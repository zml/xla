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

#include "xla/service/gpu/metal_gpu_executable.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/embed_metal_air_kernels.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/transfer_manager.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/subprocess.h"
#include "xla/types.h"

namespace xla {
namespace gpu {
namespace {

struct MatmulParams {
  uint32_t m;
  uint32_t n;
  uint32_t k;
  uint32_t reserved;
};

struct ElementwiseParams {
  uint32_t num_elements;
  uint32_t reserved0;
  uint32_t reserved1;
  uint32_t reserved2;
};

struct ReductionParams {
  uint32_t reduce_count;
  uint32_t output_elements;
  uint32_t input_minor;
  uint32_t reduction_dim;
};

struct ConvertParams {
  uint32_t num_elements;
  uint32_t reserved0;
  uint32_t reserved1;
  uint32_t reserved2;
};

enum class ReductionKind {
  kAdd,
  kMultiply,
  kMaximum,
  kMinimum,
  kAnd,
  kOr,
};

struct MetalReductionConfig {
  const HloInstruction* input = nullptr;
  PrimitiveType element_type = F32;
  int64_t parameter_number = -1;
  int64_t reduce_count = 0;
  int64_t output_elements = 1;
  int64_t input_minor = 0;
  int64_t reduction_dim = 0;
  int64_t input_dim0 = 0;
  int64_t input_dim1 = 0;
  int64_t input_dim2 = 0;
  int64_t kept_dim = 0;
  int64_t input_start = 0;
  int64_t input_stride = 1;
  float init_value = 0.0f;
  int32_t init_s32 = 0;
  bool init_pred = false;
  float output_scale = 1.0f;
  ReductionKind kind = ReductionKind::kAdd;
  bool rank2_to_rank1 = false;
  bool rank3_to_rank1 = false;
};

enum class ConvertKind {
  kF32ToS32,
  kS32ToF32,
};

struct MetalConvertConfig {
  int64_t parameter_number = 0;
  int64_t num_elements = 0;
  ConvertKind kind = ConvertKind::kF32ToS32;
};

bool IsF32Rank2Array(const Shape& shape) {
  return shape.element_type() == F32 && shape.dimensions().size() == 2;
}

bool IsF32Rank1Array(const Shape& shape) {
  return shape.element_type() == F32 && shape.dimensions().size() == 1;
}

bool IsF32Array(const Shape& shape) {
  return shape.element_type() == F32 && shape.IsArray();
}

bool IsF16Array(const Shape& shape) {
  return shape.element_type() == F16 && shape.IsArray();
}

bool IsPredArray(const Shape& shape) {
  return shape.element_type() == PRED && shape.IsArray();
}

bool IsS32Array(const Shape& shape) {
  return shape.element_type() == S32 && shape.IsArray();
}

bool IsS16Array(const Shape& shape) {
  return shape.element_type() == S16 && shape.IsArray();
}

bool IsS8Array(const Shape& shape) {
  return shape.element_type() == S8 && shape.IsArray();
}

bool IsU32Array(const Shape& shape) {
  return shape.element_type() == U32 && shape.IsArray();
}

bool IsU16Array(const Shape& shape) {
  return shape.element_type() == U16 && shape.IsArray();
}

bool IsU8Array(const Shape& shape) {
  return shape.element_type() == U8 && shape.IsArray();
}

bool IsBF16Array(const Shape& shape) {
  return shape.element_type() == BF16 && shape.IsArray();
}

bool IsPackedSubbyteArray(const Shape& shape) {
  return shape.IsArray() &&
         (shape.element_type() == S4 || shape.element_type() == U4);
}

bool IsSupportedBitcastArray(const Shape& shape) {
  switch (shape.element_type()) {
    case S4:
    case U4:
    case S8:
    case U8:
    case S16:
    case U16:
    case S32:
    case U32:
    case F16:
    case BF16:
    case F32:
      return shape.IsArray();
    default:
      return false;
  }
}

bool IsC64Array(const Shape& shape) {
  return shape.element_type() == C64 && shape.IsArray();
}

bool IsScalarLikeF32(const Shape& shape) {
  return shape.element_type() == F32 && ShapeUtil::IsEffectiveScalar(shape);
}

bool IsScalarLikeS32(const Shape& shape) {
  return shape.element_type() == S32 && ShapeUtil::IsEffectiveScalar(shape);
}

bool IsScalarLikeIndex(const Shape& shape) {
  return (shape.element_type() == S32 || shape.element_type() == S16 ||
          shape.element_type() == S8 || shape.element_type() == U32 ||
          shape.element_type() == U16 || shape.element_type() == U8) &&
         ShapeUtil::IsEffectiveScalar(shape);
}

bool IsScalarLikeSupported(const Shape& shape) {
  return (shape.element_type() == F32 || shape.element_type() == F16 ||
          shape.element_type() == BF16 || shape.element_type() == S32 ||
          shape.element_type() == S16 || shape.element_type() == S8 ||
          shape.element_type() == U32 || shape.element_type() == U16 ||
          shape.element_type() == U8 ||
          shape.element_type() == PRED) &&
         ShapeUtil::IsEffectiveScalar(shape);
}

bool IsSupportedElementwiseArray(const Shape& shape) {
  return IsF32Array(shape) || IsF16Array(shape) || IsPredArray(shape) ||
         IsS32Array(shape) || IsS16Array(shape) || IsS8Array(shape) ||
         IsU32Array(shape) || IsU16Array(shape) || IsU8Array(shape) ||
         IsBF16Array(shape);
}

bool IsOrderStatisticArray(const Shape& shape) {
  return IsF32Array(shape) || IsS32Array(shape) || IsU16Array(shape);
}

const char* ElementIrType(PrimitiveType type) {
  switch (type) {
    case S4:
    case U4:
    case S8:
    case U8:
      return "i8";
    case F32:
      return "float";
    case F16:
      return "half";
    case S32:
    case U32:
      return "i32";
    case S16:
    case U16:
    case BF16:
      return "i16";
    case PRED:
      return "i8";
    case C64:
      return "<2 x float>";
    default:
      return "float";
  }
}

const char* ValueIrType(PrimitiveType type) {
  if (type == BF16) {
    return "float";
  }
  if (type == F16) {
    return "float";
  }
  return type == PRED ? "i1" : ElementIrType(type);
}

const char* ElementAirTypeName(PrimitiveType type) {
  switch (type) {
    case S4:
    case U4:
    case U8:
      return "uchar";
    case S8:
      return "char";
    case F32:
      return "float";
    case F16:
      return "half";
    case S32:
      return "int";
    case U32:
      return "uint";
    case S16:
      return "short";
    case U16:
    case BF16:
      return "ushort";
    case PRED:
      return "bool";
    case C64:
      return "float2";
    default:
      return "float";
  }
}

int ElementTypeSize(PrimitiveType type) {
  if (type == S4 || type == U4 || type == S8 || type == U8) {
    return 1;
  }
  if (type == PRED) {
    return 1;
  }
  if (type == S16 || type == U16 || type == BF16 || type == F16) {
    return 2;
  }
  if (type == C64) {
    return 8;
  }
  return 4;
}

int ElementBitWidth(PrimitiveType type) {
  if (type == S4 || type == U4) {
    return 4;
  }
  return ElementTypeSize(type) * 8;
}

int64_t PackedByteSize(const Shape& shape) {
  return (ShapeUtil::ElementsIn(shape) * ElementBitWidth(shape.element_type()) +
          7) /
         8;
}

std::string FloatLiteral(float value) {
  double double_value = static_cast<double>(value);
  uint64_t bits = 0;
  std::memcpy(&bits, &double_value, sizeof(bits));
  return absl::StrFormat("0x%016llX",
                         static_cast<unsigned long long>(bits));
}

std::string DefaultIrValue(PrimitiveType type) {
  switch (type) {
    case F32:
    case F16:
    case BF16:
      return "0x0000000000000000";
    case PRED:
      return "false";
    case C64:
      return "zeroinitializer";
    default:
      return "0";
  }
}

bool IsScalarConvolutionElementType(PrimitiveType type) {
  switch (type) {
    case F32:
    case F16:
    case BF16:
    case S8:
    case S16:
    case S32:
    case U8:
    case U16:
    case U32:
    case PRED:
      return true;
    default:
      return false;
  }
}

bool IsFloatAccumulatorElementType(PrimitiveType type) {
  return type == F32 || type == F16 || type == BF16;
}

bool IsSignedIntegerElementType(PrimitiveType type) {
  return type == S8 || type == S16 || type == S32;
}

bool IsUnsignedIntegerElementType(PrimitiveType type) {
  return type == U8 || type == U16 || type == U32;
}

bool IsIntegerElementType(PrimitiveType type) {
  return IsSignedIntegerElementType(type) || IsUnsignedIntegerElementType(type);
}

bool IsSupportedDotElementPair(PrimitiveType input_type,
                               PrimitiveType result_type) {
  if (input_type == result_type) {
    return IsFloatAccumulatorElementType(result_type) ||
           IsIntegerElementType(result_type) || result_type == PRED ||
           result_type == C64;
  }
  if (IsFloatAccumulatorElementType(input_type) && result_type == F32) {
    return true;
  }
  if (IsIntegerElementType(input_type) &&
      IsFloatAccumulatorElementType(result_type)) {
    return true;
  }
  if (IsSignedIntegerElementType(input_type) &&
      IsSignedIntegerElementType(result_type) &&
      ElementBitWidth(input_type) <= ElementBitWidth(result_type)) {
    return true;
  }
  if (IsUnsignedIntegerElementType(input_type) &&
      IsUnsignedIntegerElementType(result_type) &&
      ElementBitWidth(input_type) <= ElementBitWidth(result_type)) {
    return true;
  }
  return false;
}

bool IsSupportedConvolutionElementPair(PrimitiveType input_type,
                                       PrimitiveType result_type) {
  if (!IsScalarConvolutionElementType(input_type) ||
      !IsScalarConvolutionElementType(result_type)) {
    return false;
  }
  if (input_type == result_type) {
    return true;
  }
  if ((input_type == F16 || input_type == BF16) && result_type == F32) {
    return true;
  }
  if (IsIntegerElementType(input_type) &&
      IsFloatAccumulatorElementType(result_type)) {
    return true;
  }
  if (IsSignedIntegerElementType(input_type) &&
      IsSignedIntegerElementType(result_type) &&
      ElementBitWidth(input_type) <= ElementBitWidth(result_type)) {
    return true;
  }
  if (IsUnsignedIntegerElementType(input_type) &&
      IsUnsignedIntegerElementType(result_type) &&
      ElementBitWidth(input_type) <= ElementBitWidth(result_type)) {
    return true;
  }
  return false;
}

bool IsZeroValue(const HloInstruction* instruction) {
  if (instruction->IsConstant() &&
      ShapeUtil::IsEffectiveScalar(instruction->shape())) {
    return instruction->literal().IsZero({});
  }
  if (instruction->opcode() == HloOpcode::kBroadcast) {
    return IsZeroValue(instruction->operand(0));
  }
  return false;
}

absl::StatusOr<const HloInstruction*> MatchDotRoot(
    const HloInstruction* root, bool* relu) {
  *relu = false;
  if (root->opcode() == HloOpcode::kDot) {
    return root;
  }
  if (root->opcode() == HloOpcode::kMaximum) {
    if (root->operand(0)->opcode() == HloOpcode::kDot &&
        IsZeroValue(root->operand(1))) {
      *relu = true;
      return root->operand(0);
    }
    if (root->operand(1)->opcode() == HloOpcode::kDot &&
        IsZeroValue(root->operand(0))) {
      *relu = true;
      return root->operand(1);
    }
  }
  return absl::UnimplementedError(
      "Metal direct AIR currently supports only f32 rank-2 dot and "
      "maximum(dot, 0).");
}

absl::StatusOr<std::string> RunCommand(std::vector<std::string> argv,
                                       bool capture_stdout) {
  if (argv.empty()) {
    return absl::InvalidArgumentError("Cannot run an empty command.");
  }
  tsl::SubProcess process;
  process.SetProgram(argv[0], argv);
  if (capture_stdout) {
    process.SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
  }
  process.SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);
  if (!process.Start()) {
    return absl::InternalError(
        absl::StrFormat("Failed to launch %s.", argv[0]));
  }

  std::string stdout_output;
  std::string stderr_output;
  int exit_status =
      process.Communicate(/*stdin_input=*/nullptr,
                          capture_stdout ? &stdout_output : nullptr,
                          &stderr_output);
  if (exit_status != 0) {
    return absl::InternalError(absl::StrFormat(
        "Command failed with status %d: %s\nstderr:\n%s", exit_status,
        absl::StrJoin(argv, " "), stderr_output));
  }
  return capture_stdout ? stdout_output : stderr_output;
}

absl::StatusOr<std::string> FindMetalTool(const char* env_name,
                                          const char* tool_name) {
  if (const char* explicit_tool = std::getenv(env_name)) {
    return std::string(explicit_tool);
  }
  if (const char* toolchain = std::getenv("METAL_TOOLCHAIN")) {
    return absl::StrCat(toolchain, "/", tool_name);
  }
  TF_ASSIGN_OR_RETURN(std::string path,
                      RunCommand({"/usr/bin/xcrun", "--find", tool_name},
                                 /*capture_stdout=*/true));
  path = std::string(absl::StripAsciiWhitespace(path));
  if (path.empty()) {
    return absl::NotFoundError(
        absl::StrFormat("xcrun could not find %s.", tool_name));
  }
  return path;
}

absl::StatusOr<std::vector<uint8_t>> CompileMetalAirToMetallib(
    absl::string_view source, absl::string_view temp_name) {
  TF_ASSIGN_OR_RETURN(std::string air_as, FindMetalTool("AIR_AS", "air-as"));
  TF_ASSIGN_OR_RETURN(std::string air_opt,
                      FindMetalTool("AIR_OPT", "air-opt"));
  TF_ASSIGN_OR_RETURN(std::string metallib,
                      FindMetalTool("METALLIB", "metallib"));

  tsl::Env* env = tsl::Env::Default();
  std::string base;
  if (!env->LocalTempFilename(&base)) {
    return absl::InternalError(
        absl::StrFormat("Could not create Metal AIR temp filename for %s.",
                        temp_name));
  }
  std::string air_ll_path = absl::StrCat(base, ".ll");
  std::string air_path = absl::StrCat(base, ".air");
  std::string opt_air_path = absl::StrCat(base, ".opt.air");
  std::string metallib_path = absl::StrCat(base, ".metallib");
  absl::Cleanup cleanup = [&] {
    env->DeleteFile(air_ll_path).IgnoreError();
    env->DeleteFile(air_path).IgnoreError();
    env->DeleteFile(opt_air_path).IgnoreError();
    env->DeleteFile(metallib_path).IgnoreError();
  };

  TF_RETURN_IF_ERROR(tsl::WriteStringToFile(env, air_ll_path, source));
  TF_RETURN_IF_ERROR(
      RunCommand({air_as, air_ll_path, "-o", air_path}, false).status());
  TF_RETURN_IF_ERROR(
      RunCommand({air_opt, "--O3", air_path, "-o", opt_air_path}, false)
          .status());
  TF_RETURN_IF_ERROR(
      RunCommand({metallib, opt_air_path, "-o", metallib_path}, false)
          .status());

  std::string bytes;
  TF_RETURN_IF_ERROR(tsl::ReadFileToString(env, metallib_path, &bytes));
  return std::vector<uint8_t>(bytes.begin(), bytes.end());
}

class ElementwiseAirEmitter {
 public:
  explicit ElementwiseAirEmitter(Shape result_shape)
      : result_shape_(std::move(result_shape)),
        num_work_items_(result_shape_.IsArray()
                            ? ShapeUtil::ElementsIn(result_shape_)
                            : 0) {}

  absl::StatusOr<std::string> Emit(const HloInstruction* root) {
    if (root->shape().IsArray() && ShapeUtil::ElementsIn(root->shape()) == 0) {
      if (!IsSupportedElementwiseArray(root->shape()) &&
          !IsC64Array(root->shape()) &&
          !(root->opcode() == HloOpcode::kBitcastConvert &&
            IsSupportedBitcastArray(root->shape()))) {
        return absl::UnimplementedError(
            "Metal direct AIR empty array emission requires a supported "
            "array element type.");
      }
      result_value_ = DefaultIrValue(root->shape().element_type());
      return BuildModule();
    }
    if (IsC64Array(root->shape())) {
      std::vector<std::string> body;
      TF_ASSIGN_OR_RETURN(std::string value, EmitComplexValue(root, &body));
      expression_body_ = std::move(body);
      result_value_ = std::move(value);
      return BuildModule();
    }
    if (!IsSupportedElementwiseArray(root->shape()) &&
        !(root->opcode() == HloOpcode::kBitcastConvert &&
          IsSupportedBitcastArray(root->shape()))) {
      return absl::UnimplementedError(
          "Metal direct AIR elementwise supports only f32, f16, bf16, s32, "
          "u16, pred, and packed subbyte bitcast-convert arrays.");
    }
    std::vector<std::string> body;
    TF_ASSIGN_OR_RETURN(std::string value,
                        EmitValue(root, /*force_scalar=*/false, &body));
    expression_body_ = std::move(body);
    result_value_ = std::move(value);
    return BuildModule();
  }

  absl::StatusOr<std::string> EmitTopKTupleElement(
      const HloInstruction* topk_instr, int64_t tuple_index) {
    if (topk_instr->opcode() != HloOpcode::kTopK ||
        !topk_instr->shape().IsTuple() || tuple_index < 0 ||
        tuple_index >= ShapeUtil::TupleElementCount(topk_instr->shape()) ||
        !ShapeUtil::Equal(topk_instr->shape().tuple_shapes(tuple_index),
                          result_shape_)) {
      return absl::UnimplementedError(
          "Metal direct AIR topk tuple emission requires a topk tuple root "
          "and a matching tuple element result shape.");
    }
    if (!IsSupportedElementwiseArray(result_shape_)) {
      return absl::UnimplementedError(
          "Metal direct AIR topk tuple emission supports only f32, s32, and "
          "pred arrays.");
    }
    std::vector<std::string> body;
    TF_ASSIGN_OR_RETURN(
        std::string value,
        EmitTopKElement(Cast<HloTopKInstruction>(topk_instr), tuple_index,
                        &body));
    expression_body_ = std::move(body);
    result_value_ = std::move(value);
    return BuildModule();
  }

  const std::vector<int64_t>& parameter_numbers() const {
    return parameter_numbers_;
  }
  int64_t num_work_items() const { return num_work_items_; }

 private:
  struct CallParameterScope {
    const HloComputation* computation = nullptr;
    absl::flat_hash_map<int64_t, const HloInstruction*> arguments;
  };

  struct ScalarParameterScope {
    const HloComputation* computation = nullptr;
    absl::flat_hash_map<int64_t, std::string> values;
  };

  absl::StatusOr<std::string> EmitValue(const HloInstruction* instr,
                                        bool force_scalar,
                                        std::vector<std::string>* body) {
    if (instr->opcode() == HloOpcode::kBroadcast) {
      if (!IsScalarLikeSupported(instr->operand(0)->shape())) {
        return EmitBroadcast(instr, body);
      }
      return EmitValue(instr->operand(0), /*force_scalar=*/true, body);
    }
    if (instr->opcode() == HloOpcode::kBitcast ||
        instr->opcode() == HloOpcode::kReshape) {
      return EmitValue(instr->operand(0), force_scalar, body);
    }
    if (instr->opcode() == HloOpcode::kSlice) {
      return EmitSlice(instr, force_scalar, body);
    }

    if (instr->IsConstant()) {
      if (!IsScalarLikeSupported(instr->shape())) {
        return absl::UnimplementedError(
            "Metal direct AIR elementwise currently supports only f32/s32 scalar "
            "constants.");
      }
      std::vector<int64_t> index(instr->shape().dimensions().size(), 0);
      if (instr->shape().element_type() == S32) {
        return absl::StrCat(instr->literal().Get<int32_t>(index));
      }
      if (instr->shape().element_type() == S16) {
        return absl::StrCat(instr->literal().Get<int16_t>(index));
      }
      if (instr->shape().element_type() == S8) {
        return absl::StrCat(
            static_cast<int>(instr->literal().Get<int8_t>(index)));
      }
      if (instr->shape().element_type() == U32) {
        return absl::StrCat(instr->literal().Get<uint32_t>(index));
      }
      if (instr->shape().element_type() == U16) {
        return absl::StrCat(instr->literal().Get<uint16_t>(index));
      }
      if (instr->shape().element_type() == U8) {
        return absl::StrCat(
            static_cast<unsigned int>(instr->literal().Get<uint8_t>(index)));
      }
      if (instr->shape().element_type() == PRED) {
        return instr->literal().Get<bool>(index) ? "true" : "false";
      }
      if (instr->shape().element_type() == F16) {
        return FloatLiteral(
            static_cast<float>(instr->literal().Get<half>(index)));
      }
      if (instr->shape().element_type() == BF16) {
        return FloatLiteral(
            static_cast<float>(instr->literal().Get<bfloat16>(index)));
      }
      return FloatLiteral(instr->literal().Get<float>(index));
    }

    if (instr->opcode() == HloOpcode::kParameter) {
      if (std::optional<std::string> override =
              ScalarParameterOverride(instr)) {
        return *override;
      }
      if (const HloInstruction* override = CallParameterOverride(instr)) {
        return EmitValue(override, force_scalar, body);
      }
      if (!force_scalar && !HasResultDimensions(instr->shape()) &&
          (!instr->shape().IsArray() ||
           ShapeUtil::ElementsIn(instr->shape()) !=
               ShapeUtil::ElementsIn(result_shape_))) {
        return absl::UnimplementedError(
          "Metal direct AIR elementwise parameters must have the result "
          "shape unless they are scalar-broadcast operands.");
      }
      const int64_t parameter_number = instr->parameter_number();
      const int input_index =
          InputIndexForParameter(parameter_number, instr->shape().element_type());
      return EmitLoad(input_index, instr->shape().element_type(),
                      force_scalar ? "0" : "%idx", body);
    }

    switch (instr->opcode()) {
      case HloOpcode::kCall:
        return EmitCall(instr, force_scalar, body);
      case HloOpcode::kClamp:
        return EmitClamp(instr, body);
      case HloOpcode::kConcatenate:
        return EmitConcatenate(instr, body);
      case HloOpcode::kConditional:
        return EmitConditional(instr, body);
      case HloOpcode::kConvolution:
        return EmitConvolution(instr, body);
      case HloOpcode::kBitcastConvert:
        return EmitBitcastConvert(instr, body);
      case HloOpcode::kConvert:
        return EmitConvert(instr, body);
      case HloOpcode::kDot:
        return EmitDot(instr, body);
      case HloOpcode::kDynamicSlice:
        return EmitDynamicSlice(instr, body);
      case HloOpcode::kGather:
        return EmitGather(instr, body);
      case HloOpcode::kGetTupleElement:
        return EmitGetTupleElement(instr, body);
      case HloOpcode::kDynamicUpdateSlice:
        return EmitDynamicUpdateSlice(instr, body);
      case HloOpcode::kIota:
        return EmitIota(instr, body);
      case HloOpcode::kPad:
        return EmitPad(instr, force_scalar, body);
      case HloOpcode::kTranspose:
        return EmitTranspose(instr, body);
      case HloOpcode::kCompare:
        return EmitCompare(instr, body);
      case HloOpcode::kReduce:
        return EmitReduce(instr, body);
      case HloOpcode::kReduceWindow:
        return EmitReduceWindow(instr, body);
      case HloOpcode::kReverse:
        return EmitReverse(instr, body);
      case HloOpcode::kScatter:
        return EmitScatter(instr, body);
      case HloOpcode::kSelect:
        return EmitSelect(instr, body);
      case HloOpcode::kSort:
        return EmitSort(instr, body);
      case HloOpcode::kAdd:
      case HloOpcode::kSubtract:
      case HloOpcode::kMultiply:
      case HloOpcode::kDivide:
      case HloOpcode::kMaximum:
      case HloOpcode::kMinimum:
      case HloOpcode::kRemainder:
        return EmitBinary(instr, body);
      case HloOpcode::kAnd:
      case HloOpcode::kOr:
      case HloOpcode::kXor:
        return EmitLogicalBinary(instr, body);
      case HloOpcode::kNot:
        return EmitNot(instr, body);
      case HloOpcode::kShiftLeft:
      case HloOpcode::kShiftRightArithmetic:
      case HloOpcode::kShiftRightLogical:
        return EmitShift(instr, body);
      case HloOpcode::kPopulationCount:
        return EmitPopcount(instr, body);
      case HloOpcode::kAtan2:
        return EmitAtan2(instr, body);
      case HloOpcode::kPower:
        return EmitIntrinsicBinary(instr, "air.fast_pow.f32", body);
      case HloOpcode::kNegate:
        return EmitNegate(instr, body);
      case HloOpcode::kAbs:
        return EmitAbs(instr, body);
      case HloOpcode::kAcos:
        return EmitIntrinsicUnary(instr, "air.fast_acos.f32", body);
      case HloOpcode::kAcosh:
        return EmitIntrinsicUnary(instr, "air.fast_acosh.f32", body);
      case HloOpcode::kAsin:
        return EmitIntrinsicUnary(instr, "air.fast_asin.f32", body);
      case HloOpcode::kAsinh:
        return EmitIntrinsicUnary(instr, "air.fast_asinh.f32", body);
      case HloOpcode::kAtanh:
        return EmitIntrinsicUnary(instr, "air.fast_atanh.f32", body);
      case HloOpcode::kCos:
        return EmitIntrinsicUnary(instr, "air.fast_cos.f32", body);
      case HloOpcode::kCosh:
        return EmitIntrinsicUnary(instr, "air.fast_cosh.f32", body);
      case HloOpcode::kExp:
        return EmitIntrinsicUnary(instr, "air.fast_exp.f32", body);
      case HloOpcode::kExpm1:
        return EmitExpm1(instr, body);
      case HloOpcode::kLog:
        return EmitIntrinsicUnary(instr, "air.fast_log.f32", body);
      case HloOpcode::kLog1p:
        return EmitLog1p(instr, body);
      case HloOpcode::kLogistic:
        return EmitLogistic(instr, body);
      case HloOpcode::kSin:
        return EmitIntrinsicUnary(instr, "air.fast_sin.f32", body);
      case HloOpcode::kSinh:
        return EmitIntrinsicUnary(instr, "air.fast_sinh.f32", body);
      case HloOpcode::kSqrt:
        return EmitIntrinsicUnary(instr, "air.fast_sqrt.f32", body);
      case HloOpcode::kTan:
        return EmitIntrinsicUnary(instr, "air.fast_tan.f32", body);
      case HloOpcode::kTanh:
        return EmitIntrinsicUnary(instr, "air.fast_tanh.f32", body);
      case HloOpcode::kRsqrt:
        return EmitIntrinsicUnary(instr, "air.fast_rsqrt.f32", body);
      case HloOpcode::kFloor:
        return EmitIntrinsicUnary(instr, "air.fast_floor.f32", body);
      case HloOpcode::kCeil:
        return EmitIntrinsicUnary(instr, "air.fast_ceil.f32", body);
      case HloOpcode::kRoundNearestEven:
        return EmitIntrinsicUnary(instr, "air.fast_rint.f32", body);
      case HloOpcode::kRoundNearestAfz:
        return EmitIntrinsicUnary(instr, "air.fast_round.f32", body);
      case HloOpcode::kSign:
        return EmitSign(instr, body);
      case HloOpcode::kErf:
        return EmitErf(instr, body);
      default:
        return absl::UnimplementedError(absl::StrFormat(
            "Metal direct AIR elementwise does not support HLO opcode %s.",
            HloOpcodeString(instr->opcode())));
    }
  }

  absl::StatusOr<std::string> EmitComplexValue(
      const HloInstruction* instr, std::vector<std::string>* body) {
    if (instr->opcode() == HloOpcode::kCall) {
      const HloComputation* callee = instr->to_apply();
      if (callee->num_parameters() != instr->operand_count()) {
        return absl::InvalidArgumentError(
            "Metal direct AIR complex call operand count does not match "
            "callee parameter count.");
      }
      CallParameterScope scope;
      scope.computation = callee;
      for (int64_t i = 0; i < instr->operand_count(); ++i) {
        scope.arguments[i] = instr->operand(i);
      }
      call_parameter_scopes_.push_back(std::move(scope));
      absl::Cleanup pop_scope = [this] { call_parameter_scopes_.pop_back(); };
      return EmitComplexValue(callee->root_instruction(), body);
    }
    if (instr->opcode() == HloOpcode::kFft) {
      return EmitRfft(instr, body);
    }
    if (instr->opcode() == HloOpcode::kConvolution) {
      if (instr->shape().dimensions().size() == 2 &&
          instr->window().dimensions().empty()) {
        return EmitComplexConvolution0D(instr, body);
      }
      return EmitComplexConvolution2D(instr, body);
    }
    if (instr->opcode() == HloOpcode::kConvert) {
      return EmitComplexConvert(instr, body);
    }
    if (instr->opcode() == HloOpcode::kDot) {
      return EmitComplexDot(instr, body);
    }
    if (instr->opcode() == HloOpcode::kBroadcast) {
      if (!IsC64Array(instr->shape())) {
        return absl::UnimplementedError(
            "Metal direct AIR complex broadcast supports only c64 arrays.");
      }
      return EmitLoadFromLinearIndex(instr, "%idx", body);
    }
    if (instr->opcode() == HloOpcode::kTranspose &&
        ShapeUtil::Equal(instr->shape(), result_shape_)) {
      if (!IsC64Array(instr->shape()) || !IsC64Array(instr->operand(0)->shape())) {
        return absl::UnimplementedError(
            "Metal direct AIR complex root transpose supports only c64 arrays.");
      }
      return EmitLoadFromLinearIndex(instr, "%idx", body);
    }
    if (instr->opcode() == HloOpcode::kParameter ||
        instr->opcode() == HloOpcode::kBitcast ||
        instr->opcode() == HloOpcode::kReshape ||
        instr->opcode() == HloOpcode::kTranspose) {
      if (!IsC64Array(instr->shape())) {
        return absl::UnimplementedError(
            "Metal direct AIR complex linear path supports only c64 arrays.");
      }
      return EmitLoadFromLinearIndex(instr, "%idx", body);
    }
    return absl::UnimplementedError(absl::StrFormat(
        "Metal direct AIR complex path does not support HLO opcode %s.",
        HloOpcodeString(instr->opcode())));
  }

  absl::StatusOr<std::string> EmitComplexConvert(
      const HloInstruction* instr, std::vector<std::string>* body) {
    const HloInstruction* operand = instr->operand(0);
    const PrimitiveType src_type = operand->shape().element_type();
    if (!IsC64Array(instr->shape()) ||
        instr->shape().dimensions() != operand->shape().dimensions()) {
      return absl::UnimplementedError(
          "Metal direct AIR complex convert requires a c64 array result with "
          "the same dimensions as the operand.");
    }
    if (src_type == C64) {
      return EmitComplexValue(operand, body);
    }
    if (src_type != PRED && src_type != F32 && src_type != F16 &&
        src_type != BF16) {
      return absl::UnimplementedError(
          "Metal direct AIR complex convert currently supports only pred and "
          "real floating-point operands.");
    }

    TF_ASSIGN_OR_RETURN(
        std::string source,
        EmitValue(operand, ShapeUtil::IsEffectiveScalar(instr->shape()), body));
    std::string real = source;
    if (src_type == PRED) {
      real = NewName("complex_convert_real");
      body->push_back(
          absl::StrFormat("  %s = uitofp i1 %s to float", real, source));
    }
    std::string real_part = NewName("complex_convert_insert_real");
    std::string complex_value = NewName("complex_convert_value");
    body->push_back(absl::StrFormat(
        "  %s = insertelement <2 x float> undef, float %s, i32 0", real_part,
        real));
    body->push_back(absl::StrFormat(
        "  %s = insertelement <2 x float> %s, float 0x0000000000000000, i32 1",
        complex_value, real_part));
    return complex_value;
  }

  absl::StatusOr<std::string> EmitComplexDot(
      const HloInstruction* instr, std::vector<std::string>* body) {
    const HloInstruction* lhs = instr->operand(0);
    const HloInstruction* rhs = instr->operand(1);
    if (!ShapeUtil::Equal(instr->shape(), result_shape_) ||
        !IsC64Array(instr->shape()) || !IsC64Array(lhs->shape()) ||
        !IsC64Array(rhs->shape())) {
      return absl::UnimplementedError(
          "Metal direct AIR complex dot requires c64 arrays matching the "
          "final result.");
    }
    const DotDimensionNumbers& dnums = instr->dot_dimension_numbers();
    if (dnums.lhs_contracting_dimensions_size() != 1 ||
        dnums.rhs_contracting_dimensions_size() != 1 ||
        dnums.lhs_batch_dimensions_size() !=
            dnums.rhs_batch_dimensions_size()) {
      return absl::UnimplementedError(
          "Metal direct AIR complex dot currently supports one contracting "
          "dimension and matching batch dimensions.");
    }

    const int64_t lhs_rank = lhs->shape().dimensions().size();
    const int64_t rhs_rank = rhs->shape().dimensions().size();
    const int64_t result_rank = instr->shape().dimensions().size();
    const int64_t lhs_contract = dnums.lhs_contracting_dimensions(0);
    const int64_t rhs_contract = dnums.rhs_contracting_dimensions(0);
    std::vector<bool> lhs_used(lhs_rank, false);
    std::vector<bool> rhs_used(rhs_rank, false);
    auto mark_dim = [](std::vector<bool>* used, int64_t dim) {
      if (dim < 0 || dim >= static_cast<int64_t>(used->size()) ||
          (*used)[dim]) {
        return false;
      }
      (*used)[dim] = true;
      return true;
    };
    if (!mark_dim(&lhs_used, lhs_contract) ||
        !mark_dim(&rhs_used, rhs_contract) ||
        lhs->shape().dimensions(lhs_contract) !=
            rhs->shape().dimensions(rhs_contract)) {
      return absl::UnimplementedError(
          "Metal direct AIR complex dot dimension numbers do not match.");
    }

    std::vector<int64_t> lhs_batch;
    std::vector<int64_t> rhs_batch;
    std::vector<int64_t> result_dims;
    result_dims.reserve(result_rank);
    for (int64_t i = 0; i < dnums.lhs_batch_dimensions_size(); ++i) {
      const int64_t lhs_dim = dnums.lhs_batch_dimensions(i);
      const int64_t rhs_dim = dnums.rhs_batch_dimensions(i);
      if (!mark_dim(&lhs_used, lhs_dim) || !mark_dim(&rhs_used, rhs_dim) ||
          lhs->shape().dimensions(lhs_dim) !=
              rhs->shape().dimensions(rhs_dim)) {
        return absl::UnimplementedError(
            "Metal direct AIR complex dot batch dimensions do not match.");
      }
      lhs_batch.push_back(lhs_dim);
      rhs_batch.push_back(rhs_dim);
      result_dims.push_back(lhs->shape().dimensions(lhs_dim));
    }

    std::vector<int64_t> lhs_outer;
    std::vector<int64_t> rhs_outer;
    for (int64_t dim = 0; dim < lhs_rank; ++dim) {
      if (!lhs_used[dim]) {
        lhs_outer.push_back(dim);
        result_dims.push_back(lhs->shape().dimensions(dim));
      }
    }
    for (int64_t dim = 0; dim < rhs_rank; ++dim) {
      if (!rhs_used[dim]) {
        rhs_outer.push_back(dim);
        result_dims.push_back(rhs->shape().dimensions(dim));
      }
    }
    if (result_dims.size() != result_rank) {
      return absl::UnimplementedError(
          "Metal direct AIR complex dot result rank does not match dot "
          "dimension numbers.");
    }
    for (int64_t dim = 0; dim < result_rank; ++dim) {
      if (instr->shape().dimensions(dim) != result_dims[dim]) {
        return absl::UnimplementedError(
            "Metal direct AIR complex dot result dimensions do not match.");
      }
    }

    auto emit_coords = [&](const Shape& shape, absl::string_view index,
                           absl::string_view prefix) {
      const int64_t rank = shape.dimensions().size();
      std::vector<std::string> coords(rank);
      std::string remaining(index);
      for (int64_t dim = 0; dim < rank; ++dim) {
        if (dim == rank - 1) {
          coords[dim] = remaining;
          break;
        }
        int64_t stride = 1;
        for (int64_t minor_dim = dim + 1; minor_dim < rank; ++minor_dim) {
          stride *= shape.dimensions(minor_dim);
        }
        std::string coord = NewName(absl::StrCat(prefix, "_coord"));
        std::string rem = NewName(absl::StrCat(prefix, "_rem"));
        body->push_back(absl::StrFormat("  %s = udiv i64 %s, %d", coord,
                                        remaining, stride));
        body->push_back(absl::StrFormat("  %s = urem i64 %s, %d", rem,
                                        remaining, stride));
        coords[dim] = coord;
        remaining = rem;
      }
      return coords;
    };
    auto emit_linear_index = [&](const Shape& shape,
                                 const std::vector<std::string>& coords,
                                 absl::string_view prefix) {
      std::string linear = "0";
      const int64_t rank = shape.dimensions().size();
      for (int64_t dim = 0; dim < rank; ++dim) {
        if (coords[dim] == "0") {
          continue;
        }
        int64_t stride = 1;
        for (int64_t minor_dim = dim + 1; minor_dim < rank; ++minor_dim) {
          stride *= shape.dimensions(minor_dim);
        }
        std::string term = coords[dim];
        if (stride != 1) {
          term = NewName(absl::StrCat(prefix, "_term"));
          body->push_back(absl::StrFormat("  %s = mul i64 %s, %d", term,
                                          coords[dim], stride));
        }
        if (linear == "0") {
          linear = term;
        } else {
          std::string sum = NewName(absl::StrCat(prefix, "_index"));
          body->push_back(
              absl::StrFormat("  %s = add i64 %s, %s", sum, linear, term));
          linear = sum;
        }
      }
      return linear;
    };

    std::vector<std::string> out_coords =
        emit_coords(instr->shape(), "%idx", "complex_dot_out");
    const int64_t k = lhs->shape().dimensions(lhs_contract);
    std::string acc_real = "0x0000000000000000";
    std::string acc_imag = "0x0000000000000000";
    for (int64_t kk = 0; kk < k; ++kk) {
      std::vector<std::string> lhs_coords(lhs_rank, "0");
      std::vector<std::string> rhs_coords(rhs_rank, "0");
      int64_t result_dim = 0;
      for (int64_t i = 0; i < lhs_batch.size(); ++i, ++result_dim) {
        lhs_coords[lhs_batch[i]] = out_coords[result_dim];
        rhs_coords[rhs_batch[i]] = out_coords[result_dim];
      }
      for (int64_t dim : lhs_outer) {
        lhs_coords[dim] = out_coords[result_dim++];
      }
      for (int64_t dim : rhs_outer) {
        rhs_coords[dim] = out_coords[result_dim++];
      }
      lhs_coords[lhs_contract] = absl::StrCat(kk);
      rhs_coords[rhs_contract] = absl::StrCat(kk);

      std::string lhs_index =
          emit_linear_index(lhs->shape(), lhs_coords, "complex_dot_lhs");
      std::string rhs_index =
          emit_linear_index(rhs->shape(), rhs_coords, "complex_dot_rhs");
      TF_ASSIGN_OR_RETURN(std::string lhs_value,
                          EmitLoadFromLinearIndex(lhs, lhs_index, body));
      TF_ASSIGN_OR_RETURN(std::string rhs_value,
                          EmitLoadFromLinearIndex(rhs, rhs_index, body));
      std::string lhs_real = NewName("complex_dot_lhs_real");
      std::string lhs_imag = NewName("complex_dot_lhs_imag");
      std::string rhs_real = NewName("complex_dot_rhs_real");
      std::string rhs_imag = NewName("complex_dot_rhs_imag");
      body->push_back(absl::StrFormat(
          "  %s = extractelement <2 x float> %s, i32 0", lhs_real,
          lhs_value));
      body->push_back(absl::StrFormat(
          "  %s = extractelement <2 x float> %s, i32 1", lhs_imag,
          lhs_value));
      body->push_back(absl::StrFormat(
          "  %s = extractelement <2 x float> %s, i32 0", rhs_real,
          rhs_value));
      body->push_back(absl::StrFormat(
          "  %s = extractelement <2 x float> %s, i32 1", rhs_imag,
          rhs_value));
      std::string rr = EmitOp("fmul fast float", lhs_real, rhs_real, body);
      std::string ii = EmitOp("fmul fast float", lhs_imag, rhs_imag, body);
      std::string ri = EmitOp("fmul fast float", lhs_real, rhs_imag, body);
      std::string ir = EmitOp("fmul fast float", lhs_imag, rhs_real, body);
      std::string product_real = EmitOp("fsub fast float", rr, ii, body);
      std::string product_imag = EmitOp("fadd fast float", ri, ir, body);
      acc_real = EmitOp("fadd fast float", acc_real, product_real, body);
      acc_imag = EmitOp("fadd fast float", acc_imag, product_imag, body);
    }

    std::string real_part = NewName("complex_dot_real");
    std::string complex_value = NewName("complex_dot_value");
    body->push_back(absl::StrFormat(
        "  %s = insertelement <2 x float> undef, float %s, i32 0", real_part,
        acc_real));
    body->push_back(absl::StrFormat(
        "  %s = insertelement <2 x float> %s, float %s, i32 1",
        complex_value, real_part, acc_imag));
    return complex_value;
  }

  absl::StatusOr<std::string> EmitRfft(const HloInstruction* instr,
                                       std::vector<std::string>* body) {
    const HloInstruction* input = instr->operand(0);
    if (instr->fft_type() != FftType::RFFT || !IsC64Array(instr->shape()) ||
        !IsF32Array(input->shape()) ||
        instr->shape().dimensions().size() != 1 ||
        input->shape().dimensions().size() != 1 ||
        instr->fft_length().size() != 1 ||
        !ShapeUtil::Equal(instr->shape(), result_shape_)) {
      return absl::UnimplementedError(
          "Metal direct AIR FFT currently supports only rank-1 f32 RFFT to "
          "c64.");
    }
    const int64_t fft_length = instr->fft_length()[0];
    if (fft_length <= 0 || input->shape().dimensions(0) != fft_length ||
        instr->shape().dimensions(0) != fft_length / 2 + 1 ||
        fft_length > 64) {
      return absl::UnimplementedError(
          "Metal direct AIR RFFT currently supports static lengths up to 64.");
    }

    const double pi = std::acos(-1.0);
    std::string selected_real;
    std::string selected_imag;
    for (int64_t k = 0; k < instr->shape().dimensions(0); ++k) {
      std::string real = "0x0000000000000000";
      std::string imag = "0x0000000000000000";
      for (int64_t n = 0; n < fft_length; ++n) {
        TF_ASSIGN_OR_RETURN(std::string sample,
                            EmitLoadFromLinearIndex(input, absl::StrCat(n),
                                                    body));
        const double angle =
            -2.0 * pi * static_cast<double>(k) * static_cast<double>(n) /
            static_cast<double>(fft_length);
        const std::string real_coeff =
            FloatLiteral(static_cast<float>(std::cos(angle)));
        const std::string imag_coeff =
            FloatLiteral(static_cast<float>(std::sin(angle)));
        std::string real_term =
            EmitOp("fmul fast float", sample, real_coeff, body);
        std::string imag_term =
            EmitOp("fmul fast float", sample, imag_coeff, body);
        real = EmitOp("fadd fast float", real, real_term, body);
        imag = EmitOp("fadd fast float", imag, imag_term, body);
      }
      if (k == 0) {
        selected_real = real;
        selected_imag = imag;
        continue;
      }
      std::string is_lane = NewName("rfft_is_lane");
      body->push_back(
          absl::StrFormat("  %s = icmp eq i64 %%idx, %d", is_lane, k));
      selected_real =
          EmitTypedSelect(F32, is_lane, real, selected_real, body, "rfft_real");
      selected_imag =
          EmitTypedSelect(F32, is_lane, imag, selected_imag, body, "rfft_imag");
    }

    std::string real_part = NewName("rfft_complex_real");
    std::string complex_value = NewName("rfft_complex");
    body->push_back(absl::StrFormat(
        "  %s = insertelement <2 x float> undef, float %s, i32 0", real_part,
        selected_real));
    body->push_back(absl::StrFormat(
        "  %s = insertelement <2 x float> %s, float %s, i32 1", complex_value,
        real_part, selected_imag));
    return complex_value;
  }

  absl::StatusOr<std::string> EmitComplexConvolution0D(
      const HloInstruction* instr, std::vector<std::string>* body) {
    const HloInstruction* lhs = instr->operand(0);
    const HloInstruction* rhs = instr->operand(1);
    const ConvolutionDimensionNumbers& dnums =
        instr->convolution_dimension_numbers();
    if (!ShapeUtil::Equal(instr->shape(), result_shape_) ||
        !IsC64Array(instr->shape()) || !IsC64Array(lhs->shape()) ||
        !IsC64Array(rhs->shape()) ||
        instr->shape().dimensions().size() != 2 ||
        lhs->shape().dimensions().size() != 2 ||
        rhs->shape().dimensions().size() != 2 ||
        dnums.input_spatial_dimensions_size() != 0 ||
        dnums.kernel_spatial_dimensions_size() != 0 ||
        dnums.output_spatial_dimensions_size() != 0 ||
        instr->window().dimensions().size() != 0 ||
        instr->feature_group_count() <= 0 || instr->batch_group_count() <= 0) {
      return absl::UnimplementedError(
          "Metal direct AIR complex 0D convolution currently supports only c64 "
          "rank-2 convolutions with positive group counts.");
    }

    auto valid_rank2_dims = [](int64_t dim0, int64_t dim1) {
      return (dim0 == 0 && dim1 == 1) || (dim0 == 1 && dim1 == 0);
    };
    if (!valid_rank2_dims(dnums.input_batch_dimension(),
                          dnums.input_feature_dimension()) ||
        !valid_rank2_dims(dnums.kernel_input_feature_dimension(),
                          dnums.kernel_output_feature_dimension()) ||
        !valid_rank2_dims(dnums.output_batch_dimension(),
                          dnums.output_feature_dimension())) {
      return absl::UnimplementedError(
          "Metal direct AIR complex 0D convolution dimension numbers must be "
          "valid rank-2 permutations.");
    }

    const int64_t input_batch_dim = dnums.input_batch_dimension();
    const int64_t input_feature_dim = dnums.input_feature_dimension();
    const int64_t kernel_input_feature_dim =
        dnums.kernel_input_feature_dimension();
    const int64_t kernel_output_feature_dim =
        dnums.kernel_output_feature_dimension();
    const int64_t output_batch_dim = dnums.output_batch_dimension();
    const int64_t output_feature_dim = dnums.output_feature_dimension();
    const int64_t feature_group_count = instr->feature_group_count();
    const int64_t batch_group_count = instr->batch_group_count();
    const int64_t input_batch = lhs->shape().dimensions(input_batch_dim);
    const int64_t input_features = lhs->shape().dimensions(input_feature_dim);
    const int64_t rhs_input_features =
        rhs->shape().dimensions(kernel_input_feature_dim);
    const int64_t output_batch = instr->shape().dimensions(output_batch_dim);
    const int64_t output_features =
        rhs->shape().dimensions(kernel_output_feature_dim);
    if (input_batch != output_batch * batch_group_count ||
        input_features != rhs_input_features * feature_group_count ||
        output_features % feature_group_count != 0 ||
        output_features % batch_group_count != 0 ||
        instr->shape().dimensions(output_feature_dim) != output_features) {
      return absl::UnimplementedError(
          "Metal direct AIR complex 0D convolution dimensions do not match.");
    }

    std::vector<std::string> out_coords(2);
    std::string out_coord0 = NewName("complex_conv0d_out_coord");
    std::string out_coord1 = NewName("complex_conv0d_out_coord");
    body->push_back(absl::StrFormat("  %s = udiv i64 %%idx, %d", out_coord0,
                                    instr->shape().dimensions(1)));
    body->push_back(absl::StrFormat("  %s = urem i64 %%idx, %d", out_coord1,
                                    instr->shape().dimensions(1)));
    out_coords[0] = out_coord0;
    out_coords[1] = out_coord1;
    std::string batch = out_coords[output_batch_dim];
    const std::string& oc = out_coords[output_feature_dim];
    if (batch_group_count != 1) {
      const int64_t out_features_per_batch_group =
          output_features / batch_group_count;
      std::string batch_group = NewName("complex_conv0d_batch_group");
      std::string batch_group_offset =
          NewName("complex_conv0d_batch_group_offset");
      std::string input_batch_coord = NewName("complex_conv0d_input_batch");
      body->push_back(absl::StrFormat("  %s = udiv i64 %s, %d", batch_group,
                                      oc, out_features_per_batch_group));
      body->push_back(absl::StrFormat("  %s = mul i64 %s, %d",
                                      batch_group_offset, batch_group,
                                      output_batch));
      body->push_back(absl::StrFormat("  %s = add i64 %s, %s",
                                      input_batch_coord, batch,
                                      batch_group_offset));
      batch = input_batch_coord;
    }
    std::string feature_group = "0";
    if (feature_group_count != 1) {
      const int64_t out_features_per_feature_group =
          output_features / feature_group_count;
      feature_group = NewName("complex_conv0d_feature_group");
      body->push_back(absl::StrFormat("  %s = udiv i64 %s, %d",
                                      feature_group, oc,
                                      out_features_per_feature_group));
    }

    auto emit_linear_index = [&](const Shape& shape,
                                 const std::vector<std::string>& coords,
                                 absl::string_view prefix) {
      std::string major_term = coords[0];
      if (shape.dimensions(1) != 1 && major_term != "0") {
        std::string scaled = NewName(absl::StrCat(prefix, "_major"));
        body->push_back(absl::StrFormat("  %s = mul i64 %s, %d", scaled,
                                        major_term, shape.dimensions(1)));
        major_term = scaled;
      }
      if (major_term == "0") {
        return coords[1];
      }
      if (coords[1] == "0") {
        return major_term;
      }
      std::string index = NewName(absl::StrCat(prefix, "_index"));
      body->push_back(
          absl::StrFormat("  %s = add i64 %s, %s", index, major_term,
                          coords[1]));
      return index;
    };

    std::string acc_real = "0x0000000000000000";
    std::string acc_imag = "0x0000000000000000";
    for (int64_t ic = 0; ic < rhs_input_features; ++ic) {
      std::string lhs_feature = absl::StrCat(ic);
      if (feature_group_count != 1) {
        std::string feature_group_offset =
            NewName("complex_conv0d_feature_group_offset");
        lhs_feature = NewName("complex_conv0d_lhs_feature");
        body->push_back(absl::StrFormat("  %s = mul i64 %s, %d",
                                        feature_group_offset, feature_group,
                                        rhs_input_features));
        body->push_back(absl::StrFormat("  %s = add i64 %s, %d", lhs_feature,
                                        feature_group_offset, ic));
      }

      std::vector<std::string> lhs_coords(2, "0");
      lhs_coords[input_batch_dim] = batch;
      lhs_coords[input_feature_dim] = lhs_feature;
      std::string lhs_index =
          emit_linear_index(lhs->shape(), lhs_coords, "complex_conv0d_lhs");

      std::vector<std::string> rhs_coords(2, "0");
      rhs_coords[kernel_input_feature_dim] = absl::StrCat(ic);
      rhs_coords[kernel_output_feature_dim] = oc;
      std::string rhs_index =
          emit_linear_index(rhs->shape(), rhs_coords, "complex_conv0d_rhs");

      TF_ASSIGN_OR_RETURN(std::string lhs_value,
                          EmitLoadFromLinearIndex(lhs, lhs_index, body));
      TF_ASSIGN_OR_RETURN(std::string rhs_value,
                          EmitLoadFromLinearIndex(rhs, rhs_index, body));
      std::string lhs_real = NewName("complex_conv0d_lhs_real");
      std::string lhs_imag = NewName("complex_conv0d_lhs_imag");
      std::string rhs_real = NewName("complex_conv0d_rhs_real");
      std::string rhs_imag = NewName("complex_conv0d_rhs_imag");
      body->push_back(absl::StrFormat(
          "  %s = extractelement <2 x float> %s, i32 0", lhs_real,
          lhs_value));
      body->push_back(absl::StrFormat(
          "  %s = extractelement <2 x float> %s, i32 1", lhs_imag,
          lhs_value));
      body->push_back(absl::StrFormat(
          "  %s = extractelement <2 x float> %s, i32 0", rhs_real,
          rhs_value));
      body->push_back(absl::StrFormat(
          "  %s = extractelement <2 x float> %s, i32 1", rhs_imag,
          rhs_value));
      std::string rr = EmitOp("fmul fast float", lhs_real, rhs_real, body);
      std::string ii = EmitOp("fmul fast float", lhs_imag, rhs_imag, body);
      std::string ri = EmitOp("fmul fast float", lhs_real, rhs_imag, body);
      std::string ir = EmitOp("fmul fast float", lhs_imag, rhs_real, body);
      std::string product_real = EmitOp("fsub fast float", rr, ii, body);
      std::string product_imag = EmitOp("fadd fast float", ri, ir, body);
      acc_real = EmitOp("fadd fast float", acc_real, product_real, body);
      acc_imag = EmitOp("fadd fast float", acc_imag, product_imag, body);
    }

    std::string real_part = NewName("complex_conv0d_real");
    std::string complex_value = NewName("complex_conv0d_value");
    body->push_back(absl::StrFormat(
        "  %s = insertelement <2 x float> undef, float %s, i32 0", real_part,
        acc_real));
    body->push_back(absl::StrFormat(
        "  %s = insertelement <2 x float> %s, float %s, i32 1", complex_value,
        real_part, acc_imag));
    return complex_value;
  }

  absl::StatusOr<std::string> EmitComplexConvolution2D(
      const HloInstruction* instr, std::vector<std::string>* body) {
    const HloInstruction* lhs = instr->operand(0);
    const HloInstruction* rhs = instr->operand(1);
    const ConvolutionDimensionNumbers& dnums =
        instr->convolution_dimension_numbers();
    if (!ShapeUtil::Equal(instr->shape(), result_shape_) ||
        !IsC64Array(instr->shape()) || !IsC64Array(lhs->shape()) ||
        !IsC64Array(rhs->shape()) ||
        instr->shape().dimensions().size() != 4 ||
        lhs->shape().dimensions().size() != 4 ||
        rhs->shape().dimensions().size() != 4 ||
        dnums.input_spatial_dimensions_size() != 2 ||
        dnums.kernel_spatial_dimensions_size() != 2 ||
        dnums.output_spatial_dimensions_size() != 2 ||
        instr->window().dimensions().size() != 2 ||
        instr->feature_group_count() <= 0 || instr->batch_group_count() <= 0) {
      return absl::UnimplementedError(
          "Metal direct AIR complex convolution currently supports only c64 "
          "rank-4 2D convolutions with positive group counts.");
    }

    auto valid_permutation = [](std::vector<int64_t> dims) {
      if (dims.size() != 4) {
        return false;
      }
      absl::c_sort(dims);
      return dims[0] == 0 && dims[1] == 1 && dims[2] == 2 && dims[3] == 3;
    };
    if (!valid_permutation({dnums.input_batch_dimension(),
                            dnums.input_feature_dimension(),
                            dnums.input_spatial_dimensions(0),
                            dnums.input_spatial_dimensions(1)}) ||
        !valid_permutation({dnums.kernel_output_feature_dimension(),
                            dnums.kernel_input_feature_dimension(),
                            dnums.kernel_spatial_dimensions(0),
                            dnums.kernel_spatial_dimensions(1)}) ||
        !valid_permutation({dnums.output_batch_dimension(),
                            dnums.output_feature_dimension(),
                            dnums.output_spatial_dimensions(0),
                            dnums.output_spatial_dimensions(1)})) {
      return absl::UnimplementedError(
          "Metal direct AIR complex convolution dimension numbers must be "
          "valid rank-4 permutations.");
    }

    const int64_t input_batch_dim = dnums.input_batch_dimension();
    const int64_t input_feature_dim = dnums.input_feature_dimension();
    const int64_t input_spatial0_dim = dnums.input_spatial_dimensions(0);
    const int64_t input_spatial1_dim = dnums.input_spatial_dimensions(1);
    const int64_t kernel_output_feature_dim =
        dnums.kernel_output_feature_dimension();
    const int64_t kernel_input_feature_dim =
        dnums.kernel_input_feature_dimension();
    const int64_t kernel_spatial0_dim = dnums.kernel_spatial_dimensions(0);
    const int64_t kernel_spatial1_dim = dnums.kernel_spatial_dimensions(1);
    const int64_t output_batch_dim = dnums.output_batch_dimension();
    const int64_t output_feature_dim = dnums.output_feature_dimension();
    const int64_t output_spatial0_dim = dnums.output_spatial_dimensions(0);
    const int64_t output_spatial1_dim = dnums.output_spatial_dimensions(1);

    const int64_t feature_group_count = instr->feature_group_count();
    const int64_t batch_group_count = instr->batch_group_count();
    const int64_t input_batch = lhs->shape().dimensions(input_batch_dim);
    const int64_t output_batch = instr->shape().dimensions(output_batch_dim);
    const int64_t in_features = lhs->shape().dimensions(input_feature_dim);
    const int64_t out_features =
        rhs->shape().dimensions(kernel_output_feature_dim);
    const int64_t rhs_input_features =
        rhs->shape().dimensions(kernel_input_feature_dim);
    const int64_t in_spatial0 = lhs->shape().dimensions(input_spatial0_dim);
    const int64_t in_spatial1 = lhs->shape().dimensions(input_spatial1_dim);
    const int64_t kernel_spatial0 =
        rhs->shape().dimensions(kernel_spatial0_dim);
    const int64_t kernel_spatial1 =
        rhs->shape().dimensions(kernel_spatial1_dim);
    const int64_t out_spatial0 =
        instr->shape().dimensions(output_spatial0_dim);
    const int64_t out_spatial1 =
        instr->shape().dimensions(output_spatial1_dim);
    if (input_batch != output_batch * batch_group_count ||
        in_features != rhs_input_features * feature_group_count ||
        out_features % feature_group_count != 0 ||
        out_features % batch_group_count != 0 ||
        instr->shape().dimensions(output_feature_dim) != out_features) {
      return absl::UnimplementedError(
          "Metal direct AIR complex convolution dimensions do not match.");
    }

    const WindowDimension& window0 = instr->window().dimensions(0);
    const WindowDimension& window1 = instr->window().dimensions(1);
    auto effective_size = [](int64_t size, int64_t dilation) {
      return size == 0 ? 0 : (size - 1) * dilation + 1;
    };
    for (const WindowDimension* dim : {&window0, &window1}) {
      if (dim->stride() <= 0 || dim->base_dilation() <= 0 ||
          dim->window_dilation() <= 0 || dim->window_reversal()) {
        return absl::UnimplementedError(
            "Metal direct AIR complex convolution does not support invalid "
            "dilation, stride, or window reversal.");
      }
    }
    const int64_t effective_input0 =
        effective_size(in_spatial0, window0.base_dilation());
    const int64_t effective_input1 =
        effective_size(in_spatial1, window1.base_dilation());
    const int64_t effective_kernel0 =
        effective_size(kernel_spatial0, window0.window_dilation());
    const int64_t effective_kernel1 =
        effective_size(kernel_spatial1, window1.window_dilation());
    const int64_t expected_out0 =
        (effective_input0 + window0.padding_low() + window0.padding_high() -
         effective_kernel0) /
            window0.stride() +
        1;
    const int64_t expected_out1 =
        (effective_input1 + window1.padding_low() + window1.padding_high() -
         effective_kernel1) /
            window1.stride() +
        1;
    if (window0.size() != kernel_spatial0 ||
        window1.size() != kernel_spatial1 || out_spatial0 != expected_out0 ||
        out_spatial1 != expected_out1) {
      return absl::UnimplementedError(
          "Metal direct AIR complex convolution window dimensions do not "
          "match.");
    }

    auto emit_coords = [&](const Shape& shape, absl::string_view index,
                           absl::string_view prefix) {
      std::vector<std::string> coords(4);
      std::string remaining(index);
      for (int64_t dim = 0; dim < 4; ++dim) {
        if (dim == 3) {
          coords[dim] = remaining;
          break;
        }
        int64_t stride = 1;
        for (int64_t minor_dim = dim + 1; minor_dim < 4; ++minor_dim) {
          stride *= shape.dimensions(minor_dim);
        }
        std::string coord = NewName(absl::StrCat(prefix, "_coord"));
        std::string rem = NewName(absl::StrCat(prefix, "_rem"));
        body->push_back(absl::StrFormat("  %s = udiv i64 %s, %d", coord,
                                        remaining, stride));
        body->push_back(absl::StrFormat("  %s = urem i64 %s, %d", rem,
                                        remaining, stride));
        coords[dim] = coord;
        remaining = rem;
      }
      return coords;
    };
    auto emit_linear_index = [&](const Shape& shape,
                                 const std::vector<std::string>& coords,
                                 absl::string_view prefix) {
      std::string linear = "0";
      for (int64_t dim = 0; dim < 4; ++dim) {
        if (coords[dim] == "0") {
          continue;
        }
        int64_t stride = 1;
        for (int64_t minor_dim = dim + 1; minor_dim < 4; ++minor_dim) {
          stride *= shape.dimensions(minor_dim);
        }
        std::string term = coords[dim];
        if (stride != 1) {
          term = NewName(absl::StrCat(prefix, "_term"));
          body->push_back(absl::StrFormat("  %s = mul i64 %s, %d", term,
                                          coords[dim], stride));
        }
        if (linear == "0") {
          linear = term;
        } else {
          std::string sum = NewName(absl::StrCat(prefix, "_index"));
          body->push_back(
              absl::StrFormat("  %s = add i64 %s, %s", sum, linear, term));
          linear = sum;
        }
      }
      return linear;
    };
    auto scaled_spatial = [&](absl::string_view output_coord,
                              const WindowDimension& window,
                              int64_t kernel_coord,
                              absl::string_view prefix) {
      std::string coord(output_coord);
      if (window.stride() != 1) {
        std::string scaled = NewName(absl::StrCat(prefix, "_scaled"));
        body->push_back(absl::StrFormat("  %s = mul i64 %s, %d", scaled,
                                        coord, window.stride()));
        coord = scaled;
      }
      const int64_t kernel_offset = kernel_coord * window.window_dilation();
      if (kernel_offset != 0) {
        std::string shifted = NewName(absl::StrCat(prefix, "_kernel"));
        body->push_back(absl::StrFormat("  %s = add i64 %s, %d", shifted,
                                        coord, kernel_offset));
        coord = shifted;
      }
      if (window.padding_low() != 0) {
        std::string padded = NewName(absl::StrCat(prefix, "_padded"));
        body->push_back(absl::StrFormat("  %s = sub i64 %s, %d", padded,
                                        coord, window.padding_low()));
        coord = padded;
      }
      return coord;
    };
    auto emit_undilated_input = [&](absl::string_view dilated_coord,
                                    int64_t effective_input,
                                    int64_t base_dilation,
                                    absl::string_view prefix) {
      std::string ge0 = NewName(absl::StrCat(prefix, "_ge0"));
      std::string lt = NewName(absl::StrCat(prefix, "_lt"));
      std::string in_range = NewName(absl::StrCat(prefix, "_in_range"));
      std::string rem = NewName(absl::StrCat(prefix, "_rem"));
      std::string divisible = NewName(absl::StrCat(prefix, "_divisible"));
      std::string ok = NewName(absl::StrCat(prefix, "_ok"));
      std::string undilated = NewName(absl::StrCat(prefix, "_undilated"));
      std::string safe = NewName(absl::StrCat(prefix, "_safe"));
      body->push_back(absl::StrFormat("  %s = icmp sge i64 %s, 0", ge0,
                                      dilated_coord));
      body->push_back(absl::StrFormat("  %s = icmp slt i64 %s, %d", lt,
                                      dilated_coord, effective_input));
      body->push_back(
          absl::StrFormat("  %s = and i1 %s, %s", in_range, ge0, lt));
      body->push_back(absl::StrFormat("  %s = srem i64 %s, %d", rem,
                                      dilated_coord, base_dilation));
      body->push_back(
          absl::StrFormat("  %s = icmp eq i64 %s, 0", divisible, rem));
      body->push_back(absl::StrFormat("  %s = and i1 %s, %s", ok, in_range,
                                      divisible));
      body->push_back(absl::StrFormat("  %s = sdiv i64 %s, %d", undilated,
                                      dilated_coord, base_dilation));
      body->push_back(absl::StrFormat(
          "  %s = select i1 %s, i64 %s, i64 0", safe, ok, undilated));
      return std::pair<std::string, std::string>(safe, ok);
    };

    std::vector<std::string> out_coords =
        emit_coords(instr->shape(), "%idx", "complex_conv_out");
    const std::string& out_batch = out_coords[output_batch_dim];
    const std::string& oc = out_coords[output_feature_dim];
    const std::string& out0 = out_coords[output_spatial0_dim];
    const std::string& out1 = out_coords[output_spatial1_dim];

    std::string input_batch_coord = out_batch;
    if (batch_group_count != 1) {
      const int64_t out_features_per_group = out_features / batch_group_count;
      std::string batch_group = NewName("complex_conv_batch_group");
      std::string batch_group_offset =
          NewName("complex_conv_batch_group_offset");
      input_batch_coord = NewName("complex_conv_input_batch");
      body->push_back(absl::StrFormat("  %s = udiv i64 %s, %d", batch_group,
                                      oc, out_features_per_group));
      body->push_back(absl::StrFormat("  %s = mul i64 %s, %d",
                                      batch_group_offset, batch_group,
                                      output_batch));
      body->push_back(absl::StrFormat("  %s = add i64 %s, %s",
                                      input_batch_coord, out_batch,
                                      batch_group_offset));
    }
    std::string feature_group = "0";
    if (feature_group_count != 1) {
      const int64_t out_features_per_feature_group =
          out_features / feature_group_count;
      feature_group = NewName("complex_conv_feature_group");
      body->push_back(absl::StrFormat("  %s = udiv i64 %s, %d",
                                      feature_group, oc,
                                      out_features_per_feature_group));
    }

    std::string acc_real = "0x0000000000000000";
    std::string acc_imag = "0x0000000000000000";
    for (int64_t k0 = 0; k0 < kernel_spatial0; ++k0) {
      for (int64_t k1 = 0; k1 < kernel_spatial1; ++k1) {
        std::string dilated0 =
            scaled_spatial(out0, window0, k0, "complex_conv_input0");
        std::string dilated1 =
            scaled_spatial(out1, window1, k1, "complex_conv_input1");
        auto [safe0, ok0] = emit_undilated_input(
            dilated0, effective_input0, window0.base_dilation(),
            "complex_conv_input0");
        auto [safe1, ok1] = emit_undilated_input(
            dilated1, effective_input1, window1.base_dilation(),
            "complex_conv_input1");
        std::string in_bounds = NewName("complex_conv_in_bounds");
        body->push_back(
            absl::StrFormat("  %s = and i1 %s, %s", in_bounds, ok0, ok1));

        for (int64_t ic = 0; ic < rhs_input_features; ++ic) {
          std::string lhs_feature = absl::StrCat(ic);
          if (feature_group_count != 1) {
            std::string feature_group_offset =
                NewName("complex_conv_feature_group_offset");
            lhs_feature = NewName("complex_conv_lhs_feature");
            body->push_back(absl::StrFormat("  %s = mul i64 %s, %d",
                                            feature_group_offset,
                                            feature_group,
                                            rhs_input_features));
            body->push_back(absl::StrFormat("  %s = add i64 %s, %d",
                                            lhs_feature, feature_group_offset,
                                            ic));
          }
          std::vector<std::string> lhs_coords(4, "0");
          lhs_coords[input_batch_dim] = input_batch_coord;
          lhs_coords[input_feature_dim] = lhs_feature;
          lhs_coords[input_spatial0_dim] = safe0;
          lhs_coords[input_spatial1_dim] = safe1;
          std::string lhs_index =
              emit_linear_index(lhs->shape(), lhs_coords, "complex_conv_lhs");

          std::vector<std::string> rhs_coords(4, "0");
          rhs_coords[kernel_output_feature_dim] = oc;
          rhs_coords[kernel_input_feature_dim] = absl::StrCat(ic);
          rhs_coords[kernel_spatial0_dim] = absl::StrCat(k0);
          rhs_coords[kernel_spatial1_dim] = absl::StrCat(k1);
          std::string rhs_index =
              emit_linear_index(rhs->shape(), rhs_coords, "complex_conv_rhs");

          TF_ASSIGN_OR_RETURN(std::string lhs_value,
                              EmitLoadFromLinearIndex(lhs, lhs_index, body));
          TF_ASSIGN_OR_RETURN(std::string rhs_value,
                              EmitLoadFromLinearIndex(rhs, rhs_index, body));
          std::string lhs_real = NewName("complex_conv_lhs_real");
          std::string lhs_imag = NewName("complex_conv_lhs_imag");
          std::string rhs_real = NewName("complex_conv_rhs_real");
          std::string rhs_imag = NewName("complex_conv_rhs_imag");
          body->push_back(absl::StrFormat(
              "  %s = extractelement <2 x float> %s, i32 0", lhs_real,
              lhs_value));
          body->push_back(absl::StrFormat(
              "  %s = extractelement <2 x float> %s, i32 1", lhs_imag,
              lhs_value));
          body->push_back(absl::StrFormat(
              "  %s = extractelement <2 x float> %s, i32 0", rhs_real,
              rhs_value));
          body->push_back(absl::StrFormat(
              "  %s = extractelement <2 x float> %s, i32 1", rhs_imag,
              rhs_value));
          std::string rr = EmitOp("fmul fast float", lhs_real, rhs_real, body);
          std::string ii = EmitOp("fmul fast float", lhs_imag, rhs_imag, body);
          std::string ri = EmitOp("fmul fast float", lhs_real, rhs_imag, body);
          std::string ir = EmitOp("fmul fast float", lhs_imag, rhs_real, body);
          std::string product_real =
              EmitOp("fsub fast float", rr, ii, body);
          std::string product_imag =
              EmitOp("fadd fast float", ri, ir, body);
          std::string contribution_real =
              NewName("complex_conv_contribution_real");
          std::string contribution_imag =
              NewName("complex_conv_contribution_imag");
          body->push_back(absl::StrFormat(
              "  %s = select i1 %s, float %s, float 0x0000000000000000",
              contribution_real, in_bounds, product_real));
          body->push_back(absl::StrFormat(
              "  %s = select i1 %s, float %s, float 0x0000000000000000",
              contribution_imag, in_bounds, product_imag));
          acc_real = EmitOp("fadd fast float", acc_real, contribution_real,
                            body);
          acc_imag = EmitOp("fadd fast float", acc_imag, contribution_imag,
                            body);
        }
      }
    }

    std::string real_part = NewName("complex_conv_real");
    std::string complex_value = NewName("complex_conv_value");
    body->push_back(absl::StrFormat(
        "  %s = insertelement <2 x float> undef, float %s, i32 0", real_part,
        acc_real));
    body->push_back(absl::StrFormat(
        "  %s = insertelement <2 x float> %s, float %s, i32 1", complex_value,
        real_part, acc_imag));
    return complex_value;
  }

  absl::StatusOr<std::string> EmitLoadFromLinearIndex(
      const HloInstruction* instr, absl::string_view index,
      std::vector<std::string>* body) {
    if (instr->IsConstant()) {
      if (!IsScalarLikeSupported(instr->shape())) {
        return absl::UnimplementedError(
            "Metal direct AIR linear load currently supports only scalar "
            "constants.");
      }
      return EmitValue(instr, /*force_scalar=*/true, body);
    }
    if (instr->opcode() == HloOpcode::kBitcast ||
        instr->opcode() == HloOpcode::kReshape) {
      if (ShapeUtil::ElementsIn(instr->shape()) !=
          ShapeUtil::ElementsIn(instr->operand(0)->shape())) {
        return absl::UnimplementedError(
            "Metal direct AIR linear load reshape must preserve element "
            "count.");
      }
      return EmitLoadFromLinearIndex(instr->operand(0), index, body);
    }
    if (instr->opcode() == HloOpcode::kTranspose) {
      TF_ASSIGN_OR_RETURN(std::string source_index,
                          EmitTransposeSourceIndex(instr, index, body));
      return EmitLoadFromLinearIndex(instr->operand(0), source_index, body);
    }
    if (instr->opcode() == HloOpcode::kBroadcast) {
      if (IsScalarLikeSupported(instr->operand(0)->shape())) {
        return EmitValue(instr->operand(0), /*force_scalar=*/true, body);
      }
      return EmitBroadcastFromLinearIndex(instr, index, body);
    }
    if (instr->opcode() == HloOpcode::kConvert) {
      const PrimitiveType src_type = instr->operand(0)->shape().element_type();
      const PrimitiveType dst_type = instr->shape().element_type();
      TF_ASSIGN_OR_RETURN(
          std::string value,
          EmitLoadFromLinearIndex(instr->operand(0), index, body));
      if (src_type == dst_type ||
          (IsFloatAccumulatorElementType(src_type) &&
           IsFloatAccumulatorElementType(dst_type))) {
        return value;
      }
      auto build_complex = [&](absl::string_view real)
          -> absl::StatusOr<std::string> {
        std::string real_part = NewName("linear_convert_complex_real");
        std::string complex_value = NewName("linear_convert_complex");
        body->push_back(absl::StrFormat(
            "  %s = insertelement <2 x float> undef, float %s, i32 0",
            real_part, real));
        body->push_back(absl::StrFormat(
            "  %s = insertelement <2 x float> %s, float "
            "0x0000000000000000, i32 1",
            complex_value, real_part));
        return complex_value;
      };
      if (dst_type == C64) {
        if (IsFloatAccumulatorElementType(src_type)) {
          return build_complex(value);
        }
        std::string real = NewName("linear_convert_complex_scalar");
        if (src_type == PRED) {
          body->push_back(
              absl::StrFormat("  %s = uitofp i1 %s to float", real, value));
          return build_complex(real);
        }
        if (IsSignedIntegerElementType(src_type)) {
          body->push_back(absl::StrFormat("  %s = sitofp %s %s to float",
                                          real, ElementIrType(src_type),
                                          value));
          return build_complex(real);
        }
        if (IsUnsignedIntegerElementType(src_type)) {
          body->push_back(absl::StrFormat("  %s = uitofp %s %s to float",
                                          real, ElementIrType(src_type),
                                          value));
          return build_complex(real);
        }
      }
      std::string converted = NewName("linear_convert");
      if (src_type == PRED && IsFloatAccumulatorElementType(dst_type)) {
        body->push_back(absl::StrFormat("  %s = uitofp i1 %s to float",
                                        converted, value));
        return converted;
      }
      if (IsSignedIntegerElementType(src_type) &&
          IsFloatAccumulatorElementType(dst_type)) {
        body->push_back(absl::StrFormat("  %s = sitofp %s %s to float",
                                        converted, ElementIrType(src_type),
                                        value));
        return converted;
      }
      if (IsUnsignedIntegerElementType(src_type) &&
          IsFloatAccumulatorElementType(dst_type)) {
        body->push_back(absl::StrFormat("  %s = uitofp %s %s to float",
                                        converted, ElementIrType(src_type),
                                        value));
        return converted;
      }
      if (src_type == PRED && IsIntegerElementType(dst_type)) {
        body->push_back(absl::StrFormat("  %s = zext i1 %s to %s", converted,
                                        value, ElementIrType(dst_type)));
        return converted;
      }
      if (IsIntegerElementType(src_type) && IsIntegerElementType(dst_type)) {
        if (ElementBitWidth(src_type) == ElementBitWidth(dst_type)) {
          return value;
        }
        if (ElementBitWidth(src_type) < ElementBitWidth(dst_type)) {
          body->push_back(absl::StrFormat(
              "  %s = %s %s %s to %s", converted,
              IsSignedIntegerElementType(src_type) ? "sext" : "zext",
              ElementIrType(src_type), value, ElementIrType(dst_type)));
          return converted;
        }
        body->push_back(absl::StrFormat("  %s = trunc %s %s to %s", converted,
                                        ElementIrType(src_type), value,
                                        ElementIrType(dst_type)));
        return converted;
      }
      if (IsSignedIntegerElementType(src_type) &&
          IsSignedIntegerElementType(dst_type) &&
          ElementBitWidth(src_type) < ElementBitWidth(dst_type)) {
        body->push_back(absl::StrFormat("  %s = sext %s %s to %s", converted,
                                        ElementIrType(src_type), value,
                                        ElementIrType(dst_type)));
        return converted;
      }
      if (IsUnsignedIntegerElementType(src_type) &&
          IsUnsignedIntegerElementType(dst_type) &&
          ElementBitWidth(src_type) < ElementBitWidth(dst_type)) {
        body->push_back(absl::StrFormat("  %s = zext %s %s to %s", converted,
                                        ElementIrType(src_type), value,
                                        ElementIrType(dst_type)));
        return converted;
      }
      return absl::UnimplementedError(absl::StrFormat(
          "Metal direct AIR linear load convert does not support %s to %s.",
          primitive_util::LowercasePrimitiveTypeName(src_type),
          primitive_util::LowercasePrimitiveTypeName(dst_type)));
    }
    if (instr->opcode() == HloOpcode::kIota) {
      const auto* iota = Cast<HloIotaInstruction>(instr);
      if ((!IsF32Array(instr->shape()) && !IsS32Array(instr->shape())) ||
          instr->shape().dimensions().size() != 1 ||
          iota->iota_dimension() != 0) {
        return absl::UnimplementedError(
            "Metal direct AIR linear load iota currently supports only rank-1 "
            "f32/s32 iotas over dimension 0.");
      }
      if (instr->shape().element_type() == S32) {
        std::string value = NewName("linear_iota_s32");
        body->push_back(
            absl::StrFormat("  %s = trunc i64 %s to i32", value, index));
        return value;
      }
      std::string value = NewName("linear_iota");
      body->push_back(
          absl::StrFormat("  %s = uitofp i64 %s to float", value, index));
      return value;
    }
    if (instr->opcode() == HloOpcode::kSlice) {
      TF_ASSIGN_OR_RETURN(std::string source_index,
                          EmitSliceSourceIndex(instr, index, body));
      return EmitLoadFromLinearIndex(instr->operand(0), source_index, body);
    }
    if (instr->opcode() == HloOpcode::kConcatenate) {
      return EmitConcatenateLinearIndex(instr, index, body);
    }
    if (instr->opcode() == HloOpcode::kDynamicSlice) {
      TF_ASSIGN_OR_RETURN(std::string source_index,
                          EmitDynamicSliceSourceIndex(instr, index, body));
      return EmitLoadFromLinearIndex(instr->operand(0), source_index, body);
    }
    if (instr->opcode() == HloOpcode::kReverse) {
      TF_ASSIGN_OR_RETURN(std::string source_index,
                          EmitReverseSourceIndex(instr, index, body));
      return EmitLoadFromLinearIndex(instr->operand(0), source_index, body);
    }
    if (instr->opcode() == HloOpcode::kNegate) {
      TF_ASSIGN_OR_RETURN(std::string value,
                          EmitLoadFromLinearIndex(instr->operand(0), index,
                                                  body));
      std::string negated = NewName("linear_neg");
      if (instr->shape().element_type() == S32) {
        body->push_back(
            absl::StrFormat("  %s = sub i32 0, %s", negated, value));
        return negated;
      }
      if (instr->shape().element_type() == F32) {
        body->push_back(
            absl::StrFormat("  %s = fneg fast float %s", negated, value));
        return negated;
      }
      return absl::UnimplementedError(
          "Metal direct AIR linear load negate currently supports only f32 "
          "and s32 arrays.");
    }
    if (instr->opcode() == HloOpcode::kCall) {
      const HloComputation* callee = instr->to_apply();
      if (callee->num_parameters() != instr->operand_count()) {
        return absl::InvalidArgumentError(
            "Metal direct AIR linear load call operand count does not match "
            "callee parameter count.");
      }
      CallParameterScope scope;
      scope.computation = callee;
      for (int64_t i = 0; i < instr->operand_count(); ++i) {
        scope.arguments[i] = instr->operand(i);
      }
      call_parameter_scopes_.push_back(std::move(scope));
      absl::Cleanup pop_scope = [this] { call_parameter_scopes_.pop_back(); };
      return EmitLoadFromLinearIndex(callee->root_instruction(), index, body);
    }
    if (instr->opcode() == HloOpcode::kSelect) {
      TF_ASSIGN_OR_RETURN(std::string pred,
                          EmitLinearOperand(instr->operand(0), index, body));
      TF_ASSIGN_OR_RETURN(std::string on_true,
                          EmitLinearOperand(instr->operand(1), index, body));
      TF_ASSIGN_OR_RETURN(std::string on_false,
                          EmitLinearOperand(instr->operand(2), index, body));
      return EmitTypedSelect(instr->shape().element_type(), pred, on_true,
                             on_false, body, "linear_select");
    }
    if (instr->opcode() == HloOpcode::kAdd ||
        instr->opcode() == HloOpcode::kSubtract) {
      TF_ASSIGN_OR_RETURN(std::string lhs,
                          EmitLinearOperand(instr->operand(0), index, body));
      TF_ASSIGN_OR_RETURN(std::string rhs,
                          EmitLinearOperand(instr->operand(1), index, body));
      if (instr->shape().element_type() == S32) {
        return EmitOp(instr->opcode() == HloOpcode::kAdd ? "add i32"
                                                         : "sub i32",
                      lhs, rhs, body);
      }
      if (instr->shape().element_type() == F32) {
        return EmitOp(instr->opcode() == HloOpcode::kAdd ? "fadd fast float"
                                                         : "fsub fast float",
                      lhs, rhs, body);
      }
      return absl::UnimplementedError(
          "Metal direct AIR linear load add/subtract currently supports only "
          "f32 and s32 arrays.");
    }
    if (instr->opcode() == HloOpcode::kCompare) {
      return EmitLinearCompare(instr, index, body);
    }
    if (instr->opcode() == HloOpcode::kAnd ||
        instr->opcode() == HloOpcode::kOr ||
        instr->opcode() == HloOpcode::kXor) {
      TF_ASSIGN_OR_RETURN(std::string lhs,
                          EmitLinearOperand(instr->operand(0), index, body));
      TF_ASSIGN_OR_RETURN(std::string rhs,
                          EmitLinearOperand(instr->operand(1), index, body));
      if (instr->shape().element_type() == PRED) {
        switch (instr->opcode()) {
          case HloOpcode::kAnd:
            return EmitOp("and i1", lhs, rhs, body);
          case HloOpcode::kOr:
            return EmitOp("or i1", lhs, rhs, body);
          case HloOpcode::kXor:
            return EmitOp("xor i1", lhs, rhs, body);
          default:
            return absl::InternalError("Unexpected pred logical HLO opcode.");
        }
      }
      if (instr->shape().element_type() == S32) {
        switch (instr->opcode()) {
          case HloOpcode::kAnd:
            return EmitOp("and i32", lhs, rhs, body);
          case HloOpcode::kOr:
            return EmitOp("or i32", lhs, rhs, body);
          case HloOpcode::kXor:
            return EmitOp("xor i32", lhs, rhs, body);
          default:
            return absl::InternalError("Unexpected s32 logical HLO opcode.");
        }
      }
      return absl::UnimplementedError(
          "Metal direct AIR linear load logical ops currently support only "
          "pred and s32 arrays.");
    }
    if (instr->opcode() == HloOpcode::kReduce) {
      if (!IsPredArray(instr->shape()) || instr->operand_count() != 2 ||
          !IsPredArray(instr->operand(0)->shape()) ||
          instr->dimensions().size() != 1 || instr->dimensions()[0] != 1 ||
          instr->operand(0)->shape().dimensions().size() != 2 ||
          instr->operand(0)->shape().dimensions(1) != 1 ||
          instr->to_apply()->root_instruction()->opcode() != HloOpcode::kAnd) {
        return absl::UnimplementedError(
            "Metal direct AIR linear load reduce currently supports only pred "
            "rank-2 degenerate reduce-and.");
      }
      return EmitLoadFromLinearIndex(instr->operand(0), index, body);
    }
    if (instr->opcode() == HloOpcode::kParameter) {
      if (const HloInstruction* override = CallParameterOverride(instr)) {
        return EmitLoadFromLinearIndex(override, index, body);
      }
      const int input_index =
          InputIndexForParameter(instr->parameter_number(),
                                 instr->shape().element_type());
      return EmitLoad(input_index, instr->shape().element_type(), index, body);
    }
    return absl::UnimplementedError(absl::StrFormat(
        "Metal direct AIR linear load does not support HLO opcode %s.",
        HloOpcodeString(instr->opcode())));
  }

  absl::StatusOr<std::string> EmitLinearOperand(
      const HloInstruction* instr, absl::string_view index,
      std::vector<std::string>* body) {
    if (ShapeUtil::IsEffectiveScalar(instr->shape())) {
      return EmitValue(instr, /*force_scalar=*/true, body);
    }
    return EmitLoadFromLinearIndex(instr, index, body);
  }

  absl::StatusOr<std::string> EmitLinearCompare(
      const HloInstruction* instr, absl::string_view index,
      std::vector<std::string>* body) {
    TF_ASSIGN_OR_RETURN(std::string lhs,
                        EmitLinearOperand(instr->operand(0), index, body));
    TF_ASSIGN_OR_RETURN(std::string rhs,
                        EmitLinearOperand(instr->operand(1), index, body));
    std::string cmp = NewName("linear_cmp");
    if (instr->operand(0)->shape().element_type() == S32) {
      absl::string_view predicate;
      switch (instr->comparison_direction()) {
        case ComparisonDirection::kEq:
          predicate = "eq";
          break;
        case ComparisonDirection::kNe:
          predicate = "ne";
          break;
        case ComparisonDirection::kGe:
          predicate = "sge";
          break;
        case ComparisonDirection::kGt:
          predicate = "sgt";
          break;
        case ComparisonDirection::kLe:
          predicate = "sle";
          break;
        case ComparisonDirection::kLt:
          predicate = "slt";
          break;
      }
      body->push_back(absl::StrFormat("  %s = icmp %s i32 %s, %s", cmp,
                                      predicate, lhs, rhs));
      return cmp;
    }
    if (instr->operand(0)->shape().element_type() == PRED) {
      absl::string_view predicate;
      switch (instr->comparison_direction()) {
        case ComparisonDirection::kEq:
          predicate = "eq";
          break;
        case ComparisonDirection::kNe:
          predicate = "ne";
          break;
        default:
          return absl::UnimplementedError(
              "Metal direct AIR linear predicate compare supports only EQ and "
              "NE.");
      }
      body->push_back(absl::StrFormat("  %s = icmp %s i1 %s, %s", cmp,
                                      predicate, lhs, rhs));
      return cmp;
    }
    if (instr->operand(0)->shape().element_type() != F32) {
      return absl::UnimplementedError(
          "Metal direct AIR linear compare supports only f32, s32, and pred "
          "operands.");
    }
    absl::string_view predicate;
    switch (instr->comparison_direction()) {
      case ComparisonDirection::kEq:
        predicate = "oeq";
        break;
      case ComparisonDirection::kNe:
        predicate = "one";
        break;
      case ComparisonDirection::kGe:
        predicate = "oge";
        break;
      case ComparisonDirection::kGt:
        predicate = "ogt";
        break;
      case ComparisonDirection::kLe:
        predicate = "ole";
        break;
      case ComparisonDirection::kLt:
        predicate = "olt";
        break;
    }
    body->push_back(absl::StrFormat("  %s = fcmp fast %s float %s, %s", cmp,
                                    predicate, lhs, rhs));
    return cmp;
  }

  absl::StatusOr<std::string> EmitSliceSourceIndex(
      const HloInstruction* instr, absl::string_view index,
      std::vector<std::string>* body) {
    const HloInstruction* operand = instr->operand(0);
    if (!IsSupportedElementwiseArray(instr->shape()) ||
        !IsSupportedElementwiseArray(operand->shape()) ||
        instr->shape().element_type() != operand->shape().element_type()) {
      return absl::UnimplementedError(
          "Metal direct AIR slice currently supports only f32/s32/pred arrays "
          "with matching element types.");
    }
    const int64_t rank = instr->shape().dimensions().size();
    if (rank != operand->shape().dimensions().size() ||
        rank != instr->slice_strides().size() || (rank != 1 && rank != 2)) {
      return absl::UnimplementedError(
          "Metal direct AIR slice currently supports only rank-1 and rank-2 "
          "array slices.");
    }
    if (rank == 1) {
      std::string source_index = std::string(index);
      const int64_t stride = instr->slice_strides(0);
      const int64_t start = instr->slice_starts(0);
      if (stride != 1) {
        std::string scaled = NewName("slice_scaled");
        body->push_back(absl::StrFormat("  %s = mul i64 %s, %d", scaled,
                                        source_index, stride));
        source_index = scaled;
      }
      if (start != 0) {
        std::string shifted = NewName("slice_idx");
        body->push_back(absl::StrFormat("  %s = add i64 %s, %d", shifted,
                                        source_index, start));
        source_index = shifted;
      }
      return source_index;
    }

    const int64_t result_minor = instr->shape().dimensions(1);
    const int64_t operand_minor = operand->shape().dimensions(1);
    std::string row = NewName("slice_row");
    std::string col = NewName("slice_col");
    body->push_back(
        absl::StrFormat("  %s = udiv i64 %s, %d", row, index, result_minor));
    body->push_back(
        absl::StrFormat("  %s = urem i64 %s, %d", col, index, result_minor));

    std::string source_row = row;
    if (instr->slice_strides(0) != 1) {
      source_row = NewName("slice_source_row_scaled");
      body->push_back(absl::StrFormat("  %s = mul i64 %s, %d", source_row,
                                      row, instr->slice_strides(0)));
    }
    if (instr->slice_starts(0) != 0) {
      std::string shifted = NewName("slice_source_row");
      body->push_back(absl::StrFormat("  %s = add i64 %s, %d", shifted,
                                      source_row, instr->slice_starts(0)));
      source_row = shifted;
    }

    std::string source_col = col;
    if (instr->slice_strides(1) != 1) {
      source_col = NewName("slice_source_col_scaled");
      body->push_back(absl::StrFormat("  %s = mul i64 %s, %d", source_col,
                                      col, instr->slice_strides(1)));
    }
    if (instr->slice_starts(1) != 0) {
      std::string shifted = NewName("slice_source_col");
      body->push_back(absl::StrFormat("  %s = add i64 %s, %d", shifted,
                                      source_col, instr->slice_starts(1)));
      source_col = shifted;
    }

    std::string source_row_offset = NewName("slice_row_offset");
    std::string source_index = NewName("slice_index");
    body->push_back(absl::StrFormat("  %s = mul i64 %s, %d",
                                    source_row_offset, source_row,
                                    operand_minor));
    body->push_back(absl::StrFormat("  %s = add i64 %s, %s", source_index,
                                    source_row_offset, source_col));
    return source_index;
  }

  bool HasReverseDimension(const HloInstruction* instr, int64_t dimension) {
    return absl::c_linear_search(instr->dimensions(), dimension);
  }

  absl::StatusOr<std::string> EmitReverseSourceIndex(
      const HloInstruction* instr, absl::string_view index,
      std::vector<std::string>* body) {
    const HloInstruction* operand = instr->operand(0);
    if (!IsSupportedElementwiseArray(instr->shape()) ||
        instr->shape().element_type() != operand->shape().element_type() ||
        !ShapeUtil::Equal(instr->shape(), operand->shape())) {
      return absl::UnimplementedError(
          "Metal direct AIR reverse currently supports only f32/s32/pred "
          "arrays with matching operand and result shapes.");
    }
    const int64_t rank = instr->shape().dimensions().size();

    std::vector<std::string> coords(rank);
    std::string remaining(index);
    for (int64_t dim = 0; dim < rank; ++dim) {
      if (dim == rank - 1) {
        coords[dim] = remaining;
        break;
      }
      int64_t stride = 1;
      for (int64_t minor_dim = dim + 1; minor_dim < rank; ++minor_dim) {
        stride *= instr->shape().dimensions(minor_dim);
      }
      std::string coord = NewName("reverse_coord");
      std::string rem = NewName("reverse_rem");
      body->push_back(absl::StrFormat("  %s = udiv i64 %s, %d", coord,
                                      remaining, stride));
      body->push_back(absl::StrFormat("  %s = urem i64 %s, %d", rem,
                                      remaining, stride));
      coords[dim] = coord;
      remaining = rem;
    }

    std::string source_index = "0";
    for (int64_t dim = 0; dim < rank; ++dim) {
      std::string coord = coords[dim];
      if (HasReverseDimension(instr, dim)) {
        std::string reversed = NewName("reverse_source_coord");
        body->push_back(absl::StrFormat("  %s = sub i64 %d, %s", reversed,
                                        instr->shape().dimensions(dim) - 1,
                                        coord));
        coord = reversed;
      }
      int64_t stride = 1;
      for (int64_t minor_dim = dim + 1; minor_dim < rank; ++minor_dim) {
        stride *= instr->shape().dimensions(minor_dim);
      }
      std::string term = coord;
      if (stride != 1) {
        term = NewName("reverse_term");
        body->push_back(absl::StrFormat("  %s = mul i64 %s, %d", term, coord,
                                        stride));
      }
      if (source_index == "0") {
        source_index = term;
      } else {
        std::string sum = NewName("reverse_index");
        body->push_back(absl::StrFormat("  %s = add i64 %s, %s", sum,
                                        source_index, term));
        source_index = sum;
      }
    }
    return source_index;
  }

  absl::StatusOr<std::string> EmitDynamicSliceSourceIndex(
      const HloInstruction* instr, absl::string_view index,
      std::vector<std::string>* body) {
    const HloInstruction* operand = instr->operand(0);
    if (!IsSupportedElementwiseArray(instr->shape()) ||
        !IsSupportedElementwiseArray(operand->shape()) ||
        instr->shape().element_type() != operand->shape().element_type() ||
        instr->shape().dimensions().size() !=
            operand->shape().dimensions().size() ||
        instr->operand_count() != instr->shape().dimensions().size() + 1) {
      return absl::UnimplementedError(
          "Metal direct AIR dynamic-slice currently supports only "
          "f32/s32/pred arrays with one start index per dimension.");
    }
    const int64_t rank = instr->shape().dimensions().size();
    if (rank != 1 && rank != 2 && rank != 3) {
      return absl::UnimplementedError(
          "Metal direct AIR dynamic-slice currently supports only rank-1, "
          "rank-2, and rank-3 arrays.");
    }

    std::vector<std::string> starts;
    starts.reserve(rank);
    for (int64_t dim = 0; dim < rank; ++dim) {
      const HloInstruction* start = instr->operand(dim + 1);
      if (!IsScalarLikeIndex(start->shape())) {
        return absl::UnimplementedError(
            "Metal direct AIR dynamic-slice requires scalar integer start "
            "indices.");
      }
      TF_ASSIGN_OR_RETURN(std::string raw, EmitStartIndexAsS32(start, body));
      const int64_t max_start =
          operand->shape().dimensions(dim) - instr->shape().dimensions(dim);
      starts.push_back(EmitClampedStartIndex(raw, max_start, body));
    }

    if (rank == 1) {
      std::string source_index = NewName("dynamic_slice_index");
      body->push_back(absl::StrFormat("  %s = add i64 %s, %s", source_index,
                                      index, starts[0]));
      return source_index;
    }

    if (rank == 2) {
      const int64_t result_minor = instr->shape().dimensions(1);
      const int64_t operand_minor = operand->shape().dimensions(1);
      std::string row = NewName("dynamic_slice_row");
      std::string col = NewName("dynamic_slice_col");
      std::string source_row = NewName("dynamic_slice_source_row");
      std::string source_col = NewName("dynamic_slice_source_col");
      std::string row_offset = NewName("dynamic_slice_row_offset");
      std::string source_index = NewName("dynamic_slice_index");
      body->push_back(
          absl::StrFormat("  %s = udiv i64 %s, %d", row, index, result_minor));
      body->push_back(
          absl::StrFormat("  %s = urem i64 %s, %d", col, index, result_minor));
      body->push_back(absl::StrFormat("  %s = add i64 %s, %s", source_row, row,
                                      starts[0]));
      body->push_back(absl::StrFormat("  %s = add i64 %s, %s", source_col, col,
                                      starts[1]));
      body->push_back(absl::StrFormat("  %s = mul i64 %s, %d", row_offset,
                                      source_row, operand_minor));
      body->push_back(absl::StrFormat("  %s = add i64 %s, %s", source_index,
                                      row_offset, source_col));
      return source_index;
    }

    const int64_t result_dim1 = instr->shape().dimensions(1);
    const int64_t result_dim2 = instr->shape().dimensions(2);
    const int64_t operand_dim1 = operand->shape().dimensions(1);
    const int64_t operand_dim2 = operand->shape().dimensions(2);
    std::string plane = NewName("dynamic_slice_plane");
    std::string dim0 = NewName("dynamic_slice_dim0");
    std::string dim1 = NewName("dynamic_slice_dim1");
    std::string dim2 = NewName("dynamic_slice_dim2");
    body->push_back(absl::StrFormat("  %s = udiv i64 %s, %d", dim0, index,
                                    result_dim1 * result_dim2));
    body->push_back(absl::StrFormat("  %s = urem i64 %s, %d", plane, index,
                                    result_dim1 * result_dim2));
    body->push_back(
        absl::StrFormat("  %s = udiv i64 %s, %d", dim1, plane, result_dim2));
    body->push_back(
        absl::StrFormat("  %s = urem i64 %s, %d", dim2, plane, result_dim2));
    std::string source_dim0 = NewName("dynamic_slice_source_dim0");
    std::string source_dim1 = NewName("dynamic_slice_source_dim1");
    std::string source_dim2 = NewName("dynamic_slice_source_dim2");
    body->push_back(absl::StrFormat("  %s = add i64 %s, %s", source_dim0,
                                    dim0, starts[0]));
    body->push_back(absl::StrFormat("  %s = add i64 %s, %s", source_dim1,
                                    dim1, starts[1]));
    body->push_back(absl::StrFormat("  %s = add i64 %s, %s", source_dim2,
                                    dim2, starts[2]));
    std::string dim0_offset = NewName("dynamic_slice_dim0_offset");
    std::string dim1_offset = NewName("dynamic_slice_dim1_offset");
    std::string base_index = NewName("dynamic_slice_base_index");
    std::string source_index = NewName("dynamic_slice_index");
    body->push_back(absl::StrFormat("  %s = mul i64 %s, %d", dim0_offset,
                                    source_dim0, operand_dim1 * operand_dim2));
    body->push_back(absl::StrFormat("  %s = mul i64 %s, %d", dim1_offset,
                                    source_dim1, operand_dim2));
    body->push_back(absl::StrFormat("  %s = add i64 %s, %s", base_index,
                                    dim0_offset, dim1_offset));
    body->push_back(absl::StrFormat("  %s = add i64 %s, %s", source_index,
                                    base_index, source_dim2));
    return source_index;
  }

  absl::StatusOr<std::string> EmitTranspose(const HloInstruction* instr,
                                            std::vector<std::string>* body) {
    const HloInstruction* operand = instr->operand(0);
    if (ShapeUtil::ElementsIn(instr->shape()) !=
            ShapeUtil::ElementsIn(result_shape_) ||
        !IsSupportedElementwiseArray(instr->shape()) ||
        !IsSupportedElementwiseArray(operand->shape()) ||
        instr->shape().element_type() != operand->shape().element_type() ||
        instr->shape().dimensions().size() !=
            operand->shape().dimensions().size() ||
        instr->dimensions().size() != instr->shape().dimensions().size()) {
      return absl::UnimplementedError(
          "Metal direct AIR root transpose requires supported arrays with "
          "matching element types, ranks, and result element count.");
    }
    return EmitLoadFromLinearIndex(instr, "%idx", body);
  }

  absl::StatusOr<std::string> EmitTransposeSourceIndex(
      const HloInstruction* instr, absl::string_view linear_index,
      std::vector<std::string>* body) {
    const HloInstruction* operand = instr->operand(0);
    auto is_supported_transpose_array = [](const Shape& shape) {
      return IsSupportedElementwiseArray(shape) || IsC64Array(shape);
    };
    if (!is_supported_transpose_array(instr->shape()) ||
        !is_supported_transpose_array(operand->shape()) ||
        instr->shape().element_type() != operand->shape().element_type() ||
        instr->shape().dimensions().size() !=
            operand->shape().dimensions().size() ||
        instr->dimensions().size() != instr->shape().dimensions().size() ||
        ShapeUtil::ElementsIn(instr->shape()) == 0) {
      return absl::UnimplementedError(
          "Metal direct AIR transpose requires supported non-empty arrays with "
          "matching element types and ranks.");
    }

    const int64_t rank = instr->shape().dimensions().size();
    if (rank == 0) {
      return "0";
    }

    std::vector<bool> seen(rank, false);
    for (int64_t result_dim = 0; result_dim < rank; ++result_dim) {
      const int64_t operand_dim = instr->dimensions()[result_dim];
      if (operand_dim < 0 || operand_dim >= rank || seen[operand_dim] ||
          instr->shape().dimensions(result_dim) !=
              operand->shape().dimensions(operand_dim)) {
        return absl::UnimplementedError(
            "Metal direct AIR transpose has an invalid permutation.");
      }
      seen[operand_dim] = true;
    }

    std::vector<int64_t> result_strides(rank, 1);
    for (int64_t dim = rank - 2; dim >= 0; --dim) {
      result_strides[dim] =
          result_strides[dim + 1] * instr->shape().dimensions(dim + 1);
    }
    std::vector<int64_t> operand_strides(rank, 1);
    for (int64_t dim = rank - 2; dim >= 0; --dim) {
      operand_strides[dim] =
          operand_strides[dim + 1] * operand->shape().dimensions(dim + 1);
    }

    std::vector<std::string> operand_coords(rank, "0");
    for (int64_t result_dim = 0; result_dim < rank; ++result_dim) {
      const int64_t result_dim_size = instr->shape().dimensions(result_dim);
      std::string coord = "0";
      if (result_dim_size != 1) {
        coord = std::string(linear_index);
        if (result_strides[result_dim] != 1) {
          coord = NewName("transpose_div");
          body->push_back(absl::StrFormat("  %s = udiv i64 %s, %d", coord,
                                          linear_index,
                                          result_strides[result_dim]));
        }
        std::string rem = NewName("transpose_coord");
        body->push_back(absl::StrFormat("  %s = urem i64 %s, %d", rem, coord,
                                        result_dim_size));
        coord = rem;
      }
      operand_coords[instr->dimensions()[result_dim]] = coord;
    }

    std::string source_index = "0";
    for (int64_t operand_dim = 0; operand_dim < rank; ++operand_dim) {
      std::string term = operand_coords[operand_dim];
      if (term != "0" && operand_strides[operand_dim] != 1) {
        term = NewName("transpose_term");
        body->push_back(absl::StrFormat("  %s = mul i64 %s, %d", term,
                                        operand_coords[operand_dim],
                                        operand_strides[operand_dim]));
      }
      if (source_index == "0") {
        source_index = term;
      } else if (term != "0") {
        std::string sum = NewName("transpose_index");
        body->push_back(absl::StrFormat("  %s = add i64 %s, %s", sum,
                                        source_index, term));
        source_index = sum;
      }
    }
    return source_index;
  }

  absl::StatusOr<std::string> EmitDot(const HloInstruction* instr,
                                      std::vector<std::string>* body) {
    const HloInstruction* lhs = instr->operand(0);
    const HloInstruction* rhs = instr->operand(1);
    const PrimitiveType dot_type = instr->shape().element_type();
    if (!IsSupportedDotElementPair(lhs->shape().element_type(), dot_type) ||
        !IsSupportedDotElementPair(rhs->shape().element_type(), dot_type) ||
        !lhs->shape().IsArray() || !rhs->shape().IsArray() ||
        !HasResultDimensions(instr->shape())) {
      return absl::UnimplementedError(
          "Metal direct AIR dot fallback currently supports only "
          "floating-point, integer, and pred arrays with supported result "
          "types.");
    }

    if (instr->shape().dimensions().size() == 2 &&
        lhs->shape().dimensions().size() == 2 &&
        rhs->shape().dimensions().size() == 2 &&
        lhs->shape().dimensions(1) == rhs->shape().dimensions(0) &&
        instr->shape().dimensions(0) == lhs->shape().dimensions(0) &&
        instr->shape().dimensions(1) == rhs->shape().dimensions(1)) {
      return EmitRank2Dot(lhs, rhs, instr->shape(), body);
    }

    const DotDimensionNumbers& dnums = instr->dot_dimension_numbers();
    if (instr->shape().dimensions().size() == 3 &&
        lhs->shape().dimensions().size() == 3 &&
        rhs->shape().dimensions().size() == 3 &&
        dnums.lhs_batch_dimensions_size() == 1 &&
        dnums.lhs_batch_dimensions(0) == 0 &&
        dnums.rhs_batch_dimensions_size() == 1 &&
        dnums.rhs_batch_dimensions(0) == 0 &&
        dnums.lhs_contracting_dimensions_size() == 1 &&
        dnums.lhs_contracting_dimensions(0) == 2 &&
        dnums.rhs_contracting_dimensions_size() == 1 &&
        dnums.rhs_contracting_dimensions(0) == 1 &&
        instr->shape().dimensions(0) == lhs->shape().dimensions(0) &&
        instr->shape().dimensions(0) == rhs->shape().dimensions(0) &&
        instr->shape().dimensions(1) == lhs->shape().dimensions(1) &&
        instr->shape().dimensions(2) == rhs->shape().dimensions(2) &&
        lhs->shape().dimensions(2) == rhs->shape().dimensions(1)) {
      const int64_t batch_count = instr->shape().dimensions(0);
      const int64_t m = instr->shape().dimensions(1);
      const int64_t n = instr->shape().dimensions(2);
      const int64_t k = lhs->shape().dimensions(2);
      std::string col = NewName("dot_col");
      std::string row_linear = NewName("dot_row_linear");
      std::string row = NewName("dot_row");
      std::string batch = NewName("dot_batch");
      body->push_back(
          absl::StrFormat("  %s = urem i64 %%idx, %d", col, n));
      body->push_back(
          absl::StrFormat("  %s = udiv i64 %%idx, %d", row_linear, n));
      body->push_back(
          absl::StrFormat("  %s = urem i64 %s, %d", row, row_linear, m));
      body->push_back(
          absl::StrFormat("  %s = udiv i64 %s, %d", batch, row_linear, m));
      return EmitDotAccumulation(lhs, rhs, batch, row, col, batch_count, m, n,
                                 k, dot_type, body);
    }

    if (instr->shape().dimensions().size() == 4 &&
        lhs->shape().dimensions().size() == 4 &&
        rhs->shape().dimensions().size() == 4 &&
        dnums.lhs_batch_dimensions_size() == 2 &&
        dnums.lhs_batch_dimensions(0) == 0 &&
        dnums.lhs_batch_dimensions(1) == 1 &&
        dnums.rhs_batch_dimensions_size() == 2 &&
        dnums.rhs_batch_dimensions(0) == 0 &&
        dnums.rhs_batch_dimensions(1) == 1 &&
        dnums.lhs_contracting_dimensions_size() == 1 &&
        dnums.lhs_contracting_dimensions(0) == 3 &&
        dnums.rhs_contracting_dimensions_size() == 1 &&
        dnums.rhs_contracting_dimensions(0) == 2 &&
        instr->shape().dimensions(0) == lhs->shape().dimensions(0) &&
        instr->shape().dimensions(0) == rhs->shape().dimensions(0) &&
        instr->shape().dimensions(1) == lhs->shape().dimensions(1) &&
        instr->shape().dimensions(1) == rhs->shape().dimensions(1) &&
        instr->shape().dimensions(2) == lhs->shape().dimensions(2) &&
        instr->shape().dimensions(3) == rhs->shape().dimensions(3) &&
        lhs->shape().dimensions(3) == rhs->shape().dimensions(2)) {
      const int64_t batch0_count = instr->shape().dimensions(0);
      const int64_t batch1_count = instr->shape().dimensions(1);
      const int64_t m = instr->shape().dimensions(2);
      const int64_t n = instr->shape().dimensions(3);
      const int64_t k = lhs->shape().dimensions(3);
      std::string col = NewName("dot4_col");
      std::string row_linear = NewName("dot4_row_linear");
      std::string row = NewName("dot4_row");
      std::string batch_linear = NewName("dot4_batch_linear");
      body->push_back(
          absl::StrFormat("  %s = urem i64 %%idx, %d", col, n));
      body->push_back(
          absl::StrFormat("  %s = udiv i64 %%idx, %d", row_linear, n));
      body->push_back(
          absl::StrFormat("  %s = urem i64 %s, %d", row, row_linear, m));
      body->push_back(
          absl::StrFormat("  %s = udiv i64 %s, %d", batch_linear,
                          row_linear, m));
      return EmitDotAccumulation(lhs, rhs, batch_linear, row, col,
                                 batch0_count * batch1_count, m, n, k,
                                 dot_type, body);
    }

    return EmitGenericDot(instr, body);
  }

  absl::StatusOr<std::string> EmitGenericDot(
      const HloInstruction* instr, std::vector<std::string>* body) {
    const HloInstruction* lhs = instr->operand(0);
    const HloInstruction* rhs = instr->operand(1);
    const DotDimensionNumbers& dnums = instr->dot_dimension_numbers();
    if (dnums.lhs_contracting_dimensions_size() !=
            dnums.rhs_contracting_dimensions_size() ||
        dnums.lhs_batch_dimensions_size() !=
            dnums.rhs_batch_dimensions_size()) {
      return absl::UnimplementedError(
          "Metal direct AIR dot fallback currently supports matching "
          "contracting and batch dimensions.");
    }

    const int64_t lhs_rank = lhs->shape().dimensions().size();
    const int64_t rhs_rank = rhs->shape().dimensions().size();
    const int64_t result_rank = instr->shape().dimensions().size();
    std::vector<bool> lhs_used(lhs_rank, false);
    std::vector<bool> rhs_used(rhs_rank, false);
    auto mark_dim = [](std::vector<bool>* used, int64_t dim) {
      if (dim < 0 || dim >= static_cast<int64_t>(used->size()) ||
          (*used)[dim]) {
        return false;
      }
      (*used)[dim] = true;
      return true;
    };

    std::vector<int64_t> lhs_contracting;
    std::vector<int64_t> rhs_contracting;
    std::vector<int64_t> contracting_sizes;
    lhs_contracting.reserve(dnums.lhs_contracting_dimensions_size());
    rhs_contracting.reserve(dnums.rhs_contracting_dimensions_size());
    contracting_sizes.reserve(dnums.lhs_contracting_dimensions_size());
    int64_t contracting_count = 1;
    for (int64_t i = 0; i < dnums.lhs_contracting_dimensions_size(); ++i) {
      const int64_t lhs_contract = dnums.lhs_contracting_dimensions(i);
      const int64_t rhs_contract = dnums.rhs_contracting_dimensions(i);
      if (!mark_dim(&lhs_used, lhs_contract) ||
          !mark_dim(&rhs_used, rhs_contract) ||
          lhs->shape().dimensions(lhs_contract) !=
              rhs->shape().dimensions(rhs_contract)) {
        return absl::UnimplementedError(
            "Metal direct AIR dot fallback contracting dimensions do not "
            "match.");
      }
      lhs_contracting.push_back(lhs_contract);
      rhs_contracting.push_back(rhs_contract);
      contracting_sizes.push_back(lhs->shape().dimensions(lhs_contract));
      contracting_count *= lhs->shape().dimensions(lhs_contract);
    }

    std::vector<int64_t> lhs_batch;
    std::vector<int64_t> rhs_batch;
    lhs_batch.reserve(dnums.lhs_batch_dimensions_size());
    rhs_batch.reserve(dnums.rhs_batch_dimensions_size());
    std::vector<int64_t> result_dims;
    result_dims.reserve(result_rank);
    for (int64_t i = 0; i < dnums.lhs_batch_dimensions_size(); ++i) {
      const int64_t lhs_dim = dnums.lhs_batch_dimensions(i);
      const int64_t rhs_dim = dnums.rhs_batch_dimensions(i);
      if (!mark_dim(&lhs_used, lhs_dim) || !mark_dim(&rhs_used, rhs_dim) ||
          lhs->shape().dimensions(lhs_dim) !=
              rhs->shape().dimensions(rhs_dim)) {
        return absl::UnimplementedError(
            "Metal direct AIR dot fallback batch dimensions do not match.");
      }
      lhs_batch.push_back(lhs_dim);
      rhs_batch.push_back(rhs_dim);
      result_dims.push_back(lhs->shape().dimensions(lhs_dim));
    }

    std::vector<int64_t> lhs_outer;
    std::vector<int64_t> rhs_outer;
    for (int64_t dim = 0; dim < lhs_rank; ++dim) {
      if (!lhs_used[dim]) {
        lhs_outer.push_back(dim);
        result_dims.push_back(lhs->shape().dimensions(dim));
      }
    }
    for (int64_t dim = 0; dim < rhs_rank; ++dim) {
      if (!rhs_used[dim]) {
        rhs_outer.push_back(dim);
        result_dims.push_back(rhs->shape().dimensions(dim));
      }
    }
    if (result_dims.size() != result_rank) {
      return absl::UnimplementedError(
          "Metal direct AIR dot fallback result rank does not match dot "
          "dimension numbers.");
    }
    for (int64_t dim = 0; dim < result_rank; ++dim) {
      if (instr->shape().dimensions(dim) != result_dims[dim]) {
        return absl::UnimplementedError(
            "Metal direct AIR dot fallback result dimensions do not match.");
      }
    }

    auto emit_coords = [&](const Shape& shape, absl::string_view index,
                           absl::string_view prefix) {
      const int64_t rank = shape.dimensions().size();
      std::vector<std::string> coords(rank);
      std::string remaining(index);
      for (int64_t dim = 0; dim < rank; ++dim) {
        if (dim == rank - 1) {
          coords[dim] = remaining;
          break;
        }
        int64_t stride = 1;
        for (int64_t minor_dim = dim + 1; minor_dim < rank; ++minor_dim) {
          stride *= shape.dimensions(minor_dim);
        }
        std::string coord = NewName(absl::StrCat(prefix, "_coord"));
        std::string rem = NewName(absl::StrCat(prefix, "_rem"));
        body->push_back(absl::StrFormat("  %s = udiv i64 %s, %d", coord,
                                        remaining, stride));
        body->push_back(absl::StrFormat("  %s = urem i64 %s, %d", rem,
                                        remaining, stride));
        coords[dim] = coord;
        remaining = rem;
      }
      return coords;
    };
    auto emit_linear_index = [&](const Shape& shape,
                                 const std::vector<std::string>& coords,
                                 absl::string_view prefix) {
      std::string linear = "0";
      const int64_t rank = shape.dimensions().size();
      for (int64_t dim = 0; dim < rank; ++dim) {
        if (coords[dim] == "0") {
          continue;
        }
        int64_t stride = 1;
        for (int64_t minor_dim = dim + 1; minor_dim < rank; ++minor_dim) {
          stride *= shape.dimensions(minor_dim);
        }
        std::string term = coords[dim];
        if (stride != 1) {
          term = NewName(absl::StrCat(prefix, "_term"));
          body->push_back(absl::StrFormat("  %s = mul i64 %s, %d", term,
                                          coords[dim], stride));
        }
        if (linear == "0") {
          linear = term;
        } else {
          std::string sum = NewName(absl::StrCat(prefix, "_index"));
          body->push_back(
              absl::StrFormat("  %s = add i64 %s, %s", sum, linear, term));
          linear = sum;
        }
      }
      return linear;
    };

    std::vector<std::string> out_coords =
        emit_coords(instr->shape(), "%idx", "dot_generic_out");
    const PrimitiveType dot_type = instr->shape().element_type();
    std::string accumulator = DefaultIrValue(dot_type);
    for (int64_t kk = 0; kk < contracting_count; ++kk) {
      std::vector<std::string> lhs_coords(lhs_rank, "0");
      std::vector<std::string> rhs_coords(rhs_rank, "0");
      int64_t result_dim = 0;
      for (int64_t i = 0; i < lhs_batch.size(); ++i, ++result_dim) {
        lhs_coords[lhs_batch[i]] = out_coords[result_dim];
        rhs_coords[rhs_batch[i]] = out_coords[result_dim];
      }
      for (int64_t dim : lhs_outer) {
        lhs_coords[dim] = out_coords[result_dim++];
      }
      for (int64_t dim : rhs_outer) {
        rhs_coords[dim] = out_coords[result_dim++];
      }
      int64_t contracting_linear = kk;
      for (int64_t i = static_cast<int64_t>(contracting_sizes.size()) - 1;
           i >= 0; --i) {
        const int64_t coord = contracting_linear % contracting_sizes[i];
        contracting_linear /= contracting_sizes[i];
        lhs_coords[lhs_contracting[i]] = absl::StrCat(coord);
        rhs_coords[rhs_contracting[i]] = absl::StrCat(coord);
      }

      std::string lhs_index =
          emit_linear_index(lhs->shape(), lhs_coords, "dot_generic_lhs");
      std::string rhs_index =
          emit_linear_index(rhs->shape(), rhs_coords, "dot_generic_rhs");
      TF_ASSIGN_OR_RETURN(std::string lhs_value,
                          EmitLoadFromLinearIndex(lhs, lhs_index, body));
      TF_ASSIGN_OR_RETURN(std::string rhs_value,
                          EmitLoadFromLinearIndex(rhs, rhs_index, body));
      TF_ASSIGN_OR_RETURN(
          lhs_value,
          EmitConvertDotOperand(lhs->shape().element_type(), dot_type,
                                lhs_value, body));
      TF_ASSIGN_OR_RETURN(
          rhs_value,
          EmitConvertDotOperand(rhs->shape().element_type(), dot_type,
                                rhs_value, body));
      TF_ASSIGN_OR_RETURN(accumulator,
                          EmitDotAccumulatorUpdate(dot_type, accumulator,
                                                   lhs_value, rhs_value, body));
    }
    return accumulator;
  }

  absl::StatusOr<std::string> EmitRank2Dot(
      const HloInstruction* lhs, const HloInstruction* rhs,
      const Shape& dot_shape, std::vector<std::string>* body) {
    const int64_t m = dot_shape.dimensions(0);
    const int64_t n = dot_shape.dimensions(1);
    const int64_t k = lhs->shape().dimensions(1);
    std::string row = NewName("dot_row");
    std::string col = NewName("dot_col");
    body->push_back(
        absl::StrFormat("  %s = udiv i64 %%idx, %d", row, n));
    body->push_back(
        absl::StrFormat("  %s = urem i64 %%idx, %d", col, n));
    return EmitDotAccumulation(lhs, rhs, /*batch=*/"0", row, col,
                               /*batch_count=*/1, m, n, k,
                               dot_shape.element_type(), body);
  }

  absl::StatusOr<std::string> EmitDotAccumulation(
      const HloInstruction* lhs, const HloInstruction* rhs,
      absl::string_view batch, absl::string_view row, absl::string_view col,
      int64_t batch_count, int64_t m, int64_t n, int64_t k,
      PrimitiveType dot_type, std::vector<std::string>* body) {
    (void)batch_count;
    std::string accumulator = DefaultIrValue(dot_type);
    for (int64_t kk = 0; kk < k; ++kk) {
      std::string lhs_index;
      std::string rhs_index;
      if (lhs->shape().dimensions().size() >= 3) {
        std::string batch_lhs_offset = NewName("dot_lhs_batch_offset");
        std::string row_lhs_offset = NewName("dot_lhs_row_offset");
        std::string lhs_base_index = NewName("dot_lhs_base_index");
        lhs_index = NewName("dot_lhs_index");
        std::string batch_rhs_offset = NewName("dot_rhs_batch_offset");
        std::string rhs_k_offset = NewName("dot_rhs_k_offset");
        rhs_index = NewName("dot_rhs_index");
        body->push_back(absl::StrFormat("  %s = mul i64 %s, %d",
                                        batch_lhs_offset, batch, m * k));
        body->push_back(absl::StrFormat("  %s = mul i64 %s, %d",
                                        row_lhs_offset, row, k));
        body->push_back(absl::StrFormat("  %s = add i64 %s, %s", lhs_base_index,
                                        batch_lhs_offset, row_lhs_offset));
        body->push_back(absl::StrFormat("  %s = add i64 %s, %d", lhs_index,
                                        lhs_base_index, kk));
        body->push_back(absl::StrFormat("  %s = mul i64 %s, %d",
                                        batch_rhs_offset, batch, k * n));
        body->push_back(absl::StrFormat("  %s = add i64 %s, %d",
                                        rhs_k_offset, batch_rhs_offset,
                                        kk * n));
        body->push_back(absl::StrFormat("  %s = add i64 %s, %s", rhs_index,
                                        rhs_k_offset, col));
      } else {
        std::string row_offset = NewName("dot_lhs_row_offset");
        lhs_index = NewName("dot_lhs_index");
        rhs_index = NewName("dot_rhs_index");
        body->push_back(
            absl::StrFormat("  %s = mul i64 %s, %d", row_offset, row, k));
        body->push_back(absl::StrFormat("  %s = add i64 %s, %d", lhs_index,
                                        row_offset, kk));
        body->push_back(absl::StrFormat("  %s = add i64 %s, %d", rhs_index,
                                        col, kk * n));
      }
      TF_ASSIGN_OR_RETURN(std::string lhs_value,
                          EmitLoadFromLinearIndex(lhs, lhs_index, body));
      TF_ASSIGN_OR_RETURN(std::string rhs_value,
                          EmitLoadFromLinearIndex(rhs, rhs_index, body));
      TF_ASSIGN_OR_RETURN(
          lhs_value,
          EmitConvertDotOperand(lhs->shape().element_type(), dot_type,
                                lhs_value, body));
      TF_ASSIGN_OR_RETURN(
          rhs_value,
          EmitConvertDotOperand(rhs->shape().element_type(), dot_type,
                                rhs_value, body));
      TF_ASSIGN_OR_RETURN(accumulator,
                          EmitDotAccumulatorUpdate(dot_type, accumulator,
                                                   lhs_value, rhs_value, body));
    }
    return accumulator;
  }

  absl::StatusOr<std::string> EmitConvertDotOperand(
      PrimitiveType src_type, PrimitiveType dot_type, absl::string_view value,
      std::vector<std::string>* body) {
    if (src_type == dot_type ||
        (IsFloatAccumulatorElementType(src_type) &&
         IsFloatAccumulatorElementType(dot_type))) {
      return std::string(value);
    }
    std::string converted = NewName("dot_operand_convert");
    if (IsIntegerElementType(src_type) &&
        IsFloatAccumulatorElementType(dot_type)) {
      body->push_back(absl::StrFormat(
          "  %s = %s %s %s to float", converted,
          IsSignedIntegerElementType(src_type) ? "sitofp" : "uitofp",
          ElementIrType(src_type), value));
      return converted;
    }
    if (IsIntegerElementType(src_type) && IsIntegerElementType(dot_type)) {
      if (ElementBitWidth(src_type) == ElementBitWidth(dot_type)) {
        return std::string(value);
      }
      if (ElementBitWidth(src_type) < ElementBitWidth(dot_type)) {
        body->push_back(absl::StrFormat(
            "  %s = %s %s %s to %s", converted,
            IsSignedIntegerElementType(src_type) ? "sext" : "zext",
            ElementIrType(src_type), value, ElementIrType(dot_type)));
        return converted;
      }
    }
    return absl::UnimplementedError(absl::StrFormat(
        "Metal direct AIR dot operand conversion does not support %s to %s.",
        primitive_util::LowercasePrimitiveTypeName(src_type),
        primitive_util::LowercasePrimitiveTypeName(dot_type)));
  }

  absl::StatusOr<std::string> EmitDotAccumulatorUpdate(
      PrimitiveType dot_type, absl::string_view accumulator,
      absl::string_view lhs_value, absl::string_view rhs_value,
      std::vector<std::string>* body) {
    if (IsFloatAccumulatorElementType(dot_type)) {
      std::string product =
          EmitOp("fmul fast float", lhs_value, rhs_value, body);
      return EmitF32ReducerOp(HloOpcode::kAdd, accumulator, product, body);
    }
    if (IsIntegerElementType(dot_type)) {
      const char* ir_type = ValueIrType(dot_type);
      std::string product =
          EmitOp(absl::StrCat("mul ", ir_type), lhs_value, rhs_value, body);
      return EmitOp(absl::StrCat("add ", ir_type), accumulator, product, body);
    }
    if (dot_type == PRED) {
      std::string product = EmitOp("and i1", lhs_value, rhs_value, body);
      return EmitOp("or i1", accumulator, product, body);
    }
    if (dot_type == C64) {
      std::string lhs_real = NewName("dot_lhs_real");
      std::string lhs_imag = NewName("dot_lhs_imag");
      std::string rhs_real = NewName("dot_rhs_real");
      std::string rhs_imag = NewName("dot_rhs_imag");
      std::string acc_real = NewName("dot_acc_real");
      std::string acc_imag = NewName("dot_acc_imag");
      body->push_back(absl::StrFormat(
          "  %s = extractelement <2 x float> %s, i32 0", lhs_real,
          lhs_value));
      body->push_back(absl::StrFormat(
          "  %s = extractelement <2 x float> %s, i32 1", lhs_imag,
          lhs_value));
      body->push_back(absl::StrFormat(
          "  %s = extractelement <2 x float> %s, i32 0", rhs_real,
          rhs_value));
      body->push_back(absl::StrFormat(
          "  %s = extractelement <2 x float> %s, i32 1", rhs_imag,
          rhs_value));
      body->push_back(absl::StrFormat(
          "  %s = extractelement <2 x float> %s, i32 0", acc_real,
          accumulator));
      body->push_back(absl::StrFormat(
          "  %s = extractelement <2 x float> %s, i32 1", acc_imag,
          accumulator));
      std::string rr = EmitOp("fmul fast float", lhs_real, rhs_real, body);
      std::string ii = EmitOp("fmul fast float", lhs_imag, rhs_imag, body);
      std::string ri = EmitOp("fmul fast float", lhs_real, rhs_imag, body);
      std::string ir = EmitOp("fmul fast float", lhs_imag, rhs_real, body);
      std::string product_real = EmitOp("fsub fast float", rr, ii, body);
      std::string product_imag = EmitOp("fadd fast float", ri, ir, body);
      std::string next_real =
          EmitOp("fadd fast float", acc_real, product_real, body);
      std::string next_imag =
          EmitOp("fadd fast float", acc_imag, product_imag, body);
      std::string real_part = NewName("dot_complex_real");
      std::string complex_value = NewName("dot_complex");
      body->push_back(absl::StrFormat(
          "  %s = insertelement <2 x float> undef, float %s, i32 0",
          real_part, next_real));
      body->push_back(absl::StrFormat(
          "  %s = insertelement <2 x float> %s, float %s, i32 1",
          complex_value, real_part, next_imag));
      return complex_value;
    }
    return absl::UnimplementedError(
        "Metal direct AIR dot accumulator does not support this element type.");
  }

  absl::StatusOr<std::string> EmitConvolution(
      const HloInstruction* instr, std::vector<std::string>* body) {
    const HloInstruction* lhs = instr->operand(0);
    const HloInstruction* rhs = instr->operand(1);
    const ConvolutionDimensionNumbers& dnums =
        instr->convolution_dimension_numbers();
    const PrimitiveType result_type = instr->shape().element_type();
    const PrimitiveType conv_type = result_type;
    const PrimitiveType lhs_type = lhs->shape().element_type();
    const PrimitiveType rhs_type = rhs->shape().element_type();
    if (lhs_type == rhs_type &&
        IsSupportedConvolutionElementPair(lhs_type, result_type) &&
        instr->shape().dimensions().size() == 4 &&
        lhs->shape().dimensions().size() == 4 &&
        rhs->shape().dimensions().size() == 4) {
      return EmitConvolution2D(instr, body);
    }
    if (IsScalarConvolutionElementType(result_type) &&
        lhs->shape().element_type() == result_type &&
        rhs->shape().element_type() == result_type &&
        instr->shape().dimensions().size() == 5 &&
        lhs->shape().dimensions().size() == 5 &&
        rhs->shape().dimensions().size() == 5) {
      return EmitConvolutionND(instr, /*spatial_rank=*/3, body);
    }
    auto is_conv0d_type = [](PrimitiveType type) {
      switch (type) {
        case F32:
        case F16:
        case BF16:
        case S8:
        case S16:
        case S32:
        case U8:
        case U16:
        case U32:
        case PRED:
          return true;
        default:
          return false;
      }
    };
    if (is_conv0d_type(conv_type) &&
        lhs->shape().element_type() == conv_type &&
        rhs->shape().element_type() == conv_type &&
        instr->shape().dimensions().size() == 2 &&
        lhs->shape().dimensions().size() == 2 &&
        rhs->shape().dimensions().size() == 2 &&
        instr->window().dimensions().empty()) {
      return EmitConvolution0D(instr, body);
    }
    auto is_conv1d_type = [](PrimitiveType type) {
      switch (type) {
        case F32:
        case F16:
        case BF16:
        case S8:
        case S16:
        case S32:
        case U8:
        case U16:
        case U32:
        case PRED:
          return true;
        default:
          return false;
      }
    };
    if (!ShapeUtil::Equal(instr->shape(), result_shape_) ||
        !is_conv1d_type(conv_type) ||
        lhs->shape().element_type() != conv_type ||
        rhs->shape().element_type() != conv_type ||
        instr->shape().dimensions().size() != 3 ||
        lhs->shape().dimensions().size() != 3 ||
        rhs->shape().dimensions().size() != 3 ||
        dnums.input_spatial_dimensions_size() != 1 ||
        dnums.kernel_spatial_dimensions_size() != 1 ||
        dnums.output_spatial_dimensions_size() != 1 ||
        instr->window().dimensions().size() != 1 ||
        instr->feature_group_count() <= 0 || instr->batch_group_count() <= 0) {
      return absl::UnimplementedError(
          "Metal direct AIR convolution currently supports only rank-3 "
          "scalar 1D convolutions with positive group counts.");
    }

    auto valid_permutation = [](std::vector<int64_t> dims) {
      if (dims.size() != 3) {
        return false;
      }
      absl::c_sort(dims);
      return dims[0] == 0 && dims[1] == 1 && dims[2] == 2;
    };
    if (!valid_permutation({dnums.input_batch_dimension(),
                            dnums.input_feature_dimension(),
                            dnums.input_spatial_dimensions(0)}) ||
        !valid_permutation({dnums.kernel_output_feature_dimension(),
                            dnums.kernel_input_feature_dimension(),
                            dnums.kernel_spatial_dimensions(0)}) ||
        !valid_permutation({dnums.output_batch_dimension(),
                            dnums.output_feature_dimension(),
                            dnums.output_spatial_dimensions(0)})) {
      return absl::UnimplementedError(
          "Metal direct AIR 1D convolution dimension numbers must be valid "
          "rank-3 permutations.");
    }

    const int64_t input_batch_dim = dnums.input_batch_dimension();
    const int64_t input_feature_dim = dnums.input_feature_dimension();
    const int64_t input_spatial_dim = dnums.input_spatial_dimensions(0);
    const int64_t kernel_output_feature_dim =
        dnums.kernel_output_feature_dimension();
    const int64_t kernel_input_feature_dim =
        dnums.kernel_input_feature_dimension();
    const int64_t kernel_spatial_dim = dnums.kernel_spatial_dimensions(0);
    const int64_t output_batch_dim = dnums.output_batch_dimension();
    const int64_t output_feature_dim = dnums.output_feature_dimension();
    const int64_t output_spatial_dim = dnums.output_spatial_dimensions(0);

    const int64_t feature_group_count = instr->feature_group_count();
    const int64_t batch_group_count = instr->batch_group_count();
    const int64_t input_batch = lhs->shape().dimensions(input_batch_dim);
    const int64_t output_batch =
        instr->shape().dimensions(output_batch_dim);
    const int64_t in_features = lhs->shape().dimensions(input_feature_dim);
    const int64_t out_features =
        rhs->shape().dimensions(kernel_output_feature_dim);
    const int64_t rhs_input_features =
        rhs->shape().dimensions(kernel_input_feature_dim);
    const int64_t in_spatial = lhs->shape().dimensions(input_spatial_dim);
    const int64_t kernel_spatial =
        rhs->shape().dimensions(kernel_spatial_dim);
    const int64_t out_spatial =
        instr->shape().dimensions(output_spatial_dim);

    const WindowDimension& dim = instr->window().dimensions(0);
    if (dim.stride() <= 0 || dim.base_dilation() <= 0 ||
        dim.window_dilation() <= 0 || dim.window_reversal()) {
      return absl::UnimplementedError(
          "Metal direct AIR convolution does not support invalid dilation, "
          "stride, or window reversal.");
    }
    auto effective_size = [](int64_t size, int64_t dilation) {
      return size == 0 ? 0 : (size - 1) * dilation + 1;
    };
    const int64_t effective_input =
        effective_size(in_spatial, dim.base_dilation());
    const int64_t effective_kernel =
        effective_size(kernel_spatial, dim.window_dilation());
    const int64_t expected_out_spatial =
        (effective_input + dim.padding_low() + dim.padding_high() -
         effective_kernel) /
            dim.stride() +
        1;
    if (input_batch != output_batch * batch_group_count ||
        in_features != rhs_input_features * feature_group_count ||
        out_features % feature_group_count != 0 ||
        out_features % batch_group_count != 0 ||
        instr->shape().dimensions(output_feature_dim) != out_features ||
        out_spatial != expected_out_spatial ||
        dim.size() != kernel_spatial) {
      return absl::UnimplementedError(
          "Metal direct AIR 1D convolution dimensions do not match.");
    }

    if (in_features == 0 || rhs_input_features == 0 || in_spatial == 0 ||
        kernel_spatial == 0) {
      return DefaultIrValue(conv_type);
    }

    auto emit_coords = [&](const Shape& shape, absl::string_view index,
                           absl::string_view prefix) {
      std::vector<std::string> coords(3);
      std::string remaining(index);
      for (int64_t dim = 0; dim < 3; ++dim) {
        if (dim == 2) {
          coords[dim] = remaining;
          break;
        }
        int64_t stride = 1;
        for (int64_t minor_dim = dim + 1; minor_dim < 3; ++minor_dim) {
          stride *= shape.dimensions(minor_dim);
        }
        std::string coord = NewName(absl::StrCat(prefix, "_coord"));
        std::string rem = NewName(absl::StrCat(prefix, "_rem"));
        body->push_back(absl::StrFormat("  %s = udiv i64 %s, %d", coord,
                                        remaining, stride));
        body->push_back(absl::StrFormat("  %s = urem i64 %s, %d", rem,
                                        remaining, stride));
        coords[dim] = coord;
        remaining = rem;
      }
      return coords;
    };
    auto emit_linear_index = [&](const Shape& shape,
                                 const std::vector<std::string>& coords,
                                 absl::string_view prefix) {
      std::string linear = "0";
      for (int64_t dim = 0; dim < 3; ++dim) {
        if (coords[dim] == "0") {
          continue;
        }
        int64_t stride = 1;
        for (int64_t minor_dim = dim + 1; minor_dim < 3; ++minor_dim) {
          stride *= shape.dimensions(minor_dim);
        }
        std::string term = coords[dim];
        if (stride != 1) {
          term = NewName(absl::StrCat(prefix, "_term"));
          body->push_back(absl::StrFormat("  %s = mul i64 %s, %d", term,
                                          coords[dim], stride));
        }
        if (linear == "0") {
          linear = term;
        } else {
          std::string sum = NewName(absl::StrCat(prefix, "_index"));
          body->push_back(
              absl::StrFormat("  %s = add i64 %s, %s", sum, linear, term));
          linear = sum;
        }
      }
      return linear;
    };
    auto scaled_spatial = [&](absl::string_view output_coord,
                              const WindowDimension& window,
                              int64_t kernel_coord,
                              absl::string_view prefix) {
      std::string coord(output_coord);
      if (window.stride() != 1) {
        std::string scaled = NewName(absl::StrCat(prefix, "_scaled"));
        body->push_back(absl::StrFormat("  %s = mul i64 %s, %d", scaled,
                                        coord, window.stride()));
        coord = scaled;
      }
      const int64_t kernel_offset = kernel_coord * window.window_dilation();
      if (kernel_offset != 0) {
        std::string shifted = NewName(absl::StrCat(prefix, "_kernel"));
        body->push_back(absl::StrFormat("  %s = add i64 %s, %d", shifted,
                                        coord, kernel_offset));
        coord = shifted;
      }
      if (window.padding_low() != 0) {
        std::string padded = NewName(absl::StrCat(prefix, "_padded"));
        body->push_back(absl::StrFormat("  %s = sub i64 %s, %d", padded,
                                        coord, window.padding_low()));
        coord = padded;
      }
      return coord;
    };
    auto emit_undilated_input = [&](absl::string_view dilated_coord,
                                    int64_t effective_input,
                                    int64_t base_dilation,
                                    absl::string_view prefix) {
      std::string ge0 = NewName(absl::StrCat(prefix, "_ge0"));
      std::string lt = NewName(absl::StrCat(prefix, "_lt"));
      std::string in_range = NewName(absl::StrCat(prefix, "_in_range"));
      std::string rem = NewName(absl::StrCat(prefix, "_rem"));
      std::string divisible = NewName(absl::StrCat(prefix, "_divisible"));
      std::string ok = NewName(absl::StrCat(prefix, "_ok"));
      std::string undilated = NewName(absl::StrCat(prefix, "_undilated"));
      std::string safe = NewName(absl::StrCat(prefix, "_safe"));
      body->push_back(absl::StrFormat("  %s = icmp sge i64 %s, 0", ge0,
                                      dilated_coord));
      body->push_back(absl::StrFormat("  %s = icmp slt i64 %s, %d", lt,
                                      dilated_coord, effective_input));
      body->push_back(
          absl::StrFormat("  %s = and i1 %s, %s", in_range, ge0, lt));
      body->push_back(absl::StrFormat("  %s = srem i64 %s, %d", rem,
                                      dilated_coord, base_dilation));
      body->push_back(
          absl::StrFormat("  %s = icmp eq i64 %s, 0", divisible, rem));
      body->push_back(absl::StrFormat("  %s = and i1 %s, %s", ok, in_range,
                                      divisible));
      body->push_back(absl::StrFormat("  %s = sdiv i64 %s, %d", undilated,
                                      dilated_coord, base_dilation));
      body->push_back(absl::StrFormat(
          "  %s = select i1 %s, i64 %s, i64 0", safe, ok, undilated));
      return std::pair<std::string, std::string>(safe, ok);
    };

    std::vector<std::string> out_coords =
        emit_coords(instr->shape(), "%idx", "conv_out");
    const std::string& out_batch = out_coords[output_batch_dim];
    const std::string& oc = out_coords[output_feature_dim];
    const std::string& out_spatial_coord = out_coords[output_spatial_dim];

    std::string input_batch_coord = out_batch;
    if (batch_group_count != 1) {
      const int64_t out_features_per_batch_group =
          out_features / batch_group_count;
      std::string batch_group = NewName("conv_batch_group");
      std::string batch_group_offset = NewName("conv_batch_group_offset");
      input_batch_coord = NewName("conv_input_batch");
      body->push_back(absl::StrFormat("  %s = udiv i64 %s, %d", batch_group,
                                      oc, out_features_per_batch_group));
      body->push_back(absl::StrFormat("  %s = mul i64 %s, %d",
                                      batch_group_offset, batch_group,
                                      output_batch));
      body->push_back(absl::StrFormat("  %s = add i64 %s, %s",
                                      input_batch_coord, out_batch,
                                      batch_group_offset));
    }
    std::string feature_group = "0";
    if (feature_group_count != 1) {
      const int64_t out_features_per_feature_group =
          out_features / feature_group_count;
      feature_group = NewName("conv_feature_group");
      body->push_back(absl::StrFormat("  %s = udiv i64 %s, %d",
                                      feature_group, oc,
                                      out_features_per_feature_group));
    }

    const bool float_accumulator =
        conv_type == F32 || conv_type == F16 || conv_type == BF16;
    const bool pred_accumulator = conv_type == PRED;
    const char* accumulator_type =
        float_accumulator ? "float" : ValueIrType(conv_type);
    std::string accumulator = DefaultIrValue(conv_type);
    for (int64_t k = 0; k < kernel_spatial; ++k) {
      std::string dilated =
          scaled_spatial(out_spatial_coord, dim, k, "conv_input");
      auto [safe_spatial, in_bounds] = emit_undilated_input(
          dilated, effective_input, dim.base_dilation(), "conv_input");
      for (int64_t ic = 0; ic < rhs_input_features; ++ic) {
        std::string lhs_feature = absl::StrCat(ic);
        if (feature_group_count != 1) {
          std::string feature_group_offset =
              NewName("conv_feature_group_offset");
          lhs_feature = NewName("conv_lhs_feature");
          body->push_back(absl::StrFormat("  %s = mul i64 %s, %d",
                                          feature_group_offset, feature_group,
                                          rhs_input_features));
          body->push_back(absl::StrFormat("  %s = add i64 %s, %d",
                                          lhs_feature, feature_group_offset,
                                          ic));
        }
        std::vector<std::string> lhs_coords(3, "0");
        lhs_coords[input_batch_dim] = input_batch_coord;
        lhs_coords[input_feature_dim] = lhs_feature;
        lhs_coords[input_spatial_dim] = safe_spatial;
        std::string lhs_index =
            emit_linear_index(lhs->shape(), lhs_coords, "conv_lhs");

        std::vector<std::string> rhs_coords(3, "0");
        rhs_coords[kernel_output_feature_dim] = oc;
        rhs_coords[kernel_input_feature_dim] = absl::StrCat(ic);
        rhs_coords[kernel_spatial_dim] = absl::StrCat(k);
        std::string rhs_index =
            emit_linear_index(rhs->shape(), rhs_coords, "conv_rhs");
        TF_ASSIGN_OR_RETURN(std::string lhs_value,
                            EmitLoadFromLinearIndex(lhs, lhs_index, body));
        TF_ASSIGN_OR_RETURN(std::string rhs_value,
                            EmitLoadFromLinearIndex(rhs, rhs_index, body));
        std::string product;
        std::string contribution = NewName("conv_contribution");
        if (float_accumulator) {
          product = EmitOp("fmul fast float", lhs_value, rhs_value, body);
          body->push_back(absl::StrFormat(
              "  %s = select i1 %s, float %s, float 0x0000000000000000",
              contribution, in_bounds, product));
          TF_ASSIGN_OR_RETURN(
              accumulator,
              EmitF32ReducerOp(HloOpcode::kAdd, accumulator, contribution,
                               body));
        } else if (pred_accumulator) {
          product = EmitOp("and i1", lhs_value, rhs_value, body);
          body->push_back(absl::StrFormat(
              "  %s = select i1 %s, i1 %s, i1 false", contribution,
              in_bounds, product));
          accumulator = EmitOp("or i1", accumulator, contribution, body);
        } else {
          product = EmitOp(absl::StrCat("mul ", accumulator_type), lhs_value,
                           rhs_value, body);
          body->push_back(absl::StrFormat(
              "  %s = select i1 %s, %s %s, %s 0", contribution, in_bounds,
              accumulator_type, product, accumulator_type));
          accumulator = EmitOp(absl::StrCat("add ", accumulator_type),
                               accumulator, contribution, body);
        }
      }
    }
    return accumulator;
  }

  absl::StatusOr<std::string> EmitConvolution0D(
      const HloInstruction* instr, std::vector<std::string>* body) {
    const HloInstruction* lhs = instr->operand(0);
    const HloInstruction* rhs = instr->operand(1);
    const ConvolutionDimensionNumbers& dnums =
        instr->convolution_dimension_numbers();
    const PrimitiveType conv_type = instr->shape().element_type();
    auto is_conv0d_type = [](PrimitiveType type) {
      switch (type) {
        case F32:
        case F16:
        case BF16:
        case S8:
        case S16:
        case S32:
        case U8:
        case U16:
        case U32:
        case PRED:
          return true;
        default:
          return false;
      }
    };
    if (!ShapeUtil::Equal(instr->shape(), result_shape_) ||
        !is_conv0d_type(conv_type) ||
        lhs->shape().element_type() != conv_type ||
        rhs->shape().element_type() != conv_type ||
        instr->shape().dimensions().size() != 2 ||
        lhs->shape().dimensions().size() != 2 ||
        rhs->shape().dimensions().size() != 2 ||
        dnums.input_spatial_dimensions_size() != 0 ||
        dnums.kernel_spatial_dimensions_size() != 0 ||
        dnums.output_spatial_dimensions_size() != 0 ||
        instr->window().dimensions().size() != 0 ||
        instr->feature_group_count() <= 0 || instr->batch_group_count() <= 0) {
      return absl::UnimplementedError(
          "Metal direct AIR 0D convolution currently supports only scalar "
          "rank-2 convolutions with positive group counts.");
    }

    auto valid_rank2_dims = [](int64_t dim0, int64_t dim1) {
      return (dim0 == 0 && dim1 == 1) || (dim0 == 1 && dim1 == 0);
    };
    if (!valid_rank2_dims(dnums.input_batch_dimension(),
                          dnums.input_feature_dimension()) ||
        !valid_rank2_dims(dnums.kernel_input_feature_dimension(),
                          dnums.kernel_output_feature_dimension()) ||
        !valid_rank2_dims(dnums.output_batch_dimension(),
                          dnums.output_feature_dimension())) {
      return absl::UnimplementedError(
          "Metal direct AIR 0D convolution dimension numbers must be valid "
          "rank-2 permutations.");
    }

    const int64_t input_batch_dim = dnums.input_batch_dimension();
    const int64_t input_feature_dim = dnums.input_feature_dimension();
    const int64_t kernel_input_feature_dim =
        dnums.kernel_input_feature_dimension();
    const int64_t kernel_output_feature_dim =
        dnums.kernel_output_feature_dimension();
    const int64_t output_batch_dim = dnums.output_batch_dimension();
    const int64_t output_feature_dim = dnums.output_feature_dimension();
    const int64_t feature_group_count = instr->feature_group_count();
    const int64_t batch_group_count = instr->batch_group_count();
    const int64_t batch_count = lhs->shape().dimensions(input_batch_dim);
    const int64_t in_features = lhs->shape().dimensions(input_feature_dim);
    const int64_t out_features =
        rhs->shape().dimensions(kernel_output_feature_dim);
    const int64_t rhs_input_features =
        rhs->shape().dimensions(kernel_input_feature_dim);
    const int64_t output_batch = instr->shape().dimensions(output_batch_dim);
    if (batch_count != output_batch * batch_group_count ||
        in_features != rhs_input_features * feature_group_count ||
        out_features % feature_group_count != 0 ||
        out_features % batch_group_count != 0 ||
        instr->shape().dimensions(output_feature_dim) != out_features) {
      return absl::UnimplementedError(
          "Metal direct AIR 0D convolution dimensions do not match.");
    }

    std::vector<std::string> out_coords(2);
    std::string out_coord0 = NewName("conv0d_out_coord");
    std::string out_coord1 = NewName("conv0d_out_coord");
    body->push_back(absl::StrFormat("  %s = udiv i64 %%idx, %d", out_coord0,
                                    instr->shape().dimensions(1)));
    body->push_back(absl::StrFormat("  %s = urem i64 %%idx, %d", out_coord1,
                                    instr->shape().dimensions(1)));
    out_coords[0] = out_coord0;
    out_coords[1] = out_coord1;
    std::string batch = out_coords[output_batch_dim];
    const std::string& oc = out_coords[output_feature_dim];
    if (batch_group_count != 1) {
      const int64_t out_features_per_batch_group =
          out_features / batch_group_count;
      std::string batch_group = NewName("conv0d_batch_group");
      std::string batch_group_offset = NewName("conv0d_batch_group_offset");
      std::string input_batch = NewName("conv0d_input_batch");
      body->push_back(absl::StrFormat("  %s = udiv i64 %s, %d", batch_group,
                                      oc, out_features_per_batch_group));
      body->push_back(absl::StrFormat("  %s = mul i64 %s, %d",
                                      batch_group_offset, batch_group,
                                      output_batch));
      body->push_back(absl::StrFormat("  %s = add i64 %s, %s", input_batch,
                                      batch, batch_group_offset));
      batch = input_batch;
    }
    std::string feature_group = "0";
    if (feature_group_count != 1) {
      const int64_t out_features_per_feature_group =
          out_features / feature_group_count;
      feature_group = NewName("conv0d_feature_group");
      body->push_back(absl::StrFormat("  %s = udiv i64 %s, %d",
                                      feature_group, oc,
                                      out_features_per_feature_group));
    }

    auto emit_linear_index = [&](const Shape& shape,
                                 const std::vector<std::string>& coords,
                                 absl::string_view prefix) {
      std::string major_term = coords[0];
      if (shape.dimensions(1) != 1 && major_term != "0") {
        std::string scaled = NewName(absl::StrCat(prefix, "_major"));
        body->push_back(absl::StrFormat("  %s = mul i64 %s, %d", scaled,
                                        major_term, shape.dimensions(1)));
        major_term = scaled;
      }
      if (major_term == "0") {
        return coords[1];
      }
      if (coords[1] == "0") {
        return major_term;
      }
      std::string index = NewName(absl::StrCat(prefix, "_index"));
      body->push_back(
          absl::StrFormat("  %s = add i64 %s, %s", index, major_term,
                          coords[1]));
      return index;
    };

    const bool float_accumulator =
        conv_type == F32 || conv_type == F16 || conv_type == BF16;
    const bool pred_accumulator = conv_type == PRED;
    const char* accumulator_type =
        float_accumulator ? "float" : ValueIrType(conv_type);
    std::string accumulator = float_accumulator   ? "0x0000000000000000"
                              : pred_accumulator ? "false"
                                                 : "0";
    for (int64_t ic = 0; ic < rhs_input_features; ++ic) {
      std::string lhs_feature = absl::StrCat(ic);
      if (feature_group_count != 1) {
        std::string feature_group_offset =
            NewName("conv0d_feature_group_offset");
        lhs_feature = NewName("conv0d_lhs_feature");
        body->push_back(absl::StrFormat("  %s = mul i64 %s, %d",
                                        feature_group_offset, feature_group,
                                        rhs_input_features));
        body->push_back(absl::StrFormat("  %s = add i64 %s, %d", lhs_feature,
                                        feature_group_offset, ic));
      }
      std::vector<std::string> lhs_coords(2, "0");
      lhs_coords[input_batch_dim] = batch;
      lhs_coords[input_feature_dim] = lhs_feature;
      std::string lhs_index =
          emit_linear_index(lhs->shape(), lhs_coords, "conv0d_lhs");

      std::vector<std::string> rhs_coords(2, "0");
      rhs_coords[kernel_input_feature_dim] = absl::StrCat(ic);
      rhs_coords[kernel_output_feature_dim] = oc;
      std::string rhs_index =
          emit_linear_index(rhs->shape(), rhs_coords, "conv0d_rhs");

      TF_ASSIGN_OR_RETURN(std::string lhs_value,
                          EmitLoadFromLinearIndex(lhs, lhs_index, body));
      TF_ASSIGN_OR_RETURN(std::string rhs_value,
                          EmitLoadFromLinearIndex(rhs, rhs_index, body));
      std::string product;
      if (float_accumulator) {
        product = EmitOp("fmul fast float", lhs_value, rhs_value, body);
        accumulator =
            EmitOp("fadd fast float", accumulator, product, body);
      } else if (pred_accumulator) {
        product = EmitOp("and i1", lhs_value, rhs_value, body);
        accumulator = EmitOp("or i1", accumulator, product, body);
      } else {
        product = EmitOp(absl::StrCat("mul ", accumulator_type), lhs_value,
                         rhs_value, body);
        accumulator = EmitOp(absl::StrCat("add ", accumulator_type),
                             accumulator, product, body);
      }
    }
    return accumulator;
  }

  absl::StatusOr<std::string> EmitConvolutionCastValue(
      absl::string_view value, PrimitiveType input_type,
      PrimitiveType result_type, std::vector<std::string>* body) {
    if (input_type == result_type ||
        ((input_type == F16 || input_type == BF16) && result_type == F32)) {
      return std::string(value);
    }
    if (IsFloatAccumulatorElementType(result_type)) {
      std::string converted = NewName("conv_cast");
      if (IsSignedIntegerElementType(input_type)) {
        body->push_back(absl::StrFormat("  %s = sitofp %s %s to float",
                                        converted, ElementIrType(input_type),
                                        value));
        return converted;
      }
      if (IsUnsignedIntegerElementType(input_type)) {
        body->push_back(absl::StrFormat("  %s = uitofp %s %s to float",
                                        converted, ElementIrType(input_type),
                                        value));
        return converted;
      }
    }
    if (IsSignedIntegerElementType(input_type) &&
        IsSignedIntegerElementType(result_type) &&
        ElementBitWidth(input_type) < ElementBitWidth(result_type)) {
      std::string converted = NewName("conv_cast");
      body->push_back(absl::StrFormat("  %s = sext %s %s to %s", converted,
                                      ElementIrType(input_type), value,
                                      ElementIrType(result_type)));
      return converted;
    }
    if (IsUnsignedIntegerElementType(input_type) &&
        IsUnsignedIntegerElementType(result_type) &&
        ElementBitWidth(input_type) < ElementBitWidth(result_type)) {
      std::string converted = NewName("conv_cast");
      body->push_back(absl::StrFormat("  %s = zext %s %s to %s", converted,
                                      ElementIrType(input_type), value,
                                      ElementIrType(result_type)));
      return converted;
    }
    return absl::UnimplementedError(absl::StrFormat(
        "Metal direct AIR convolution does not support %s inputs with %s "
        "results.",
        primitive_util::LowercasePrimitiveTypeName(input_type),
        primitive_util::LowercasePrimitiveTypeName(result_type)));
  }

  absl::StatusOr<std::string> EmitConvolution2D(
      const HloInstruction* instr, std::vector<std::string>* body) {
    const HloInstruction* lhs = instr->operand(0);
    const HloInstruction* rhs = instr->operand(1);
    const ConvolutionDimensionNumbers& dnums =
        instr->convolution_dimension_numbers();
    const PrimitiveType result_type = instr->shape().element_type();
    const PrimitiveType input_type = lhs->shape().element_type();
    if (!ShapeUtil::Equal(instr->shape(), result_shape_) ||
        !IsSupportedConvolutionElementPair(input_type, result_type) ||
        rhs->shape().element_type() != input_type ||
        instr->shape().dimensions().size() != 4 ||
        lhs->shape().dimensions().size() != 4 ||
        rhs->shape().dimensions().size() != 4 ||
        dnums.input_spatial_dimensions_size() != 2 ||
        dnums.kernel_spatial_dimensions_size() != 2 ||
        dnums.output_spatial_dimensions_size() != 2 ||
        instr->window().dimensions().size() != 2 ||
        instr->feature_group_count() <= 0 || instr->batch_group_count() <= 0) {
      return absl::UnimplementedError(
          "Metal direct AIR convolution currently supports only rank-4 "
          "scalar 2D convolutions with positive group counts.");
    }

    auto valid_permutation = [](std::vector<int64_t> dims) {
      if (dims.size() != 4) {
        return false;
      }
      absl::c_sort(dims);
      return dims[0] == 0 && dims[1] == 1 && dims[2] == 2 && dims[3] == 3;
    };
    if (!valid_permutation({dnums.input_batch_dimension(),
                            dnums.input_feature_dimension(),
                            dnums.input_spatial_dimensions(0),
                            dnums.input_spatial_dimensions(1)}) ||
        !valid_permutation({dnums.kernel_output_feature_dimension(),
                            dnums.kernel_input_feature_dimension(),
                            dnums.kernel_spatial_dimensions(0),
                            dnums.kernel_spatial_dimensions(1)}) ||
        !valid_permutation({dnums.output_batch_dimension(),
                            dnums.output_feature_dimension(),
                            dnums.output_spatial_dimensions(0),
                            dnums.output_spatial_dimensions(1)})) {
      return absl::UnimplementedError(
          "Metal direct AIR 2D convolution dimension numbers must be valid "
          "rank-4 permutations.");
    }

    const int64_t input_batch_dim = dnums.input_batch_dimension();
    const int64_t input_feature_dim = dnums.input_feature_dimension();
    const int64_t input_spatial0_dim = dnums.input_spatial_dimensions(0);
    const int64_t input_spatial1_dim = dnums.input_spatial_dimensions(1);
    const int64_t kernel_output_feature_dim =
        dnums.kernel_output_feature_dimension();
    const int64_t kernel_input_feature_dim =
        dnums.kernel_input_feature_dimension();
    const int64_t kernel_spatial0_dim = dnums.kernel_spatial_dimensions(0);
    const int64_t kernel_spatial1_dim = dnums.kernel_spatial_dimensions(1);
    const int64_t output_batch_dim = dnums.output_batch_dimension();
    const int64_t output_feature_dim = dnums.output_feature_dimension();
    const int64_t output_spatial0_dim = dnums.output_spatial_dimensions(0);
    const int64_t output_spatial1_dim = dnums.output_spatial_dimensions(1);

    const int64_t feature_group_count = instr->feature_group_count();
    const int64_t batch_group_count = instr->batch_group_count();
    const int64_t input_batch = lhs->shape().dimensions(input_batch_dim);
    const int64_t output_batch =
        instr->shape().dimensions(output_batch_dim);
    const int64_t in_features = lhs->shape().dimensions(input_feature_dim);
    const int64_t out_features =
        rhs->shape().dimensions(kernel_output_feature_dim);
    const int64_t rhs_input_features =
        rhs->shape().dimensions(kernel_input_feature_dim);
    const int64_t in_spatial0 = lhs->shape().dimensions(input_spatial0_dim);
    const int64_t in_spatial1 = lhs->shape().dimensions(input_spatial1_dim);
    const int64_t kernel_spatial0 =
        rhs->shape().dimensions(kernel_spatial0_dim);
    const int64_t kernel_spatial1 =
        rhs->shape().dimensions(kernel_spatial1_dim);
    const int64_t out_spatial0 =
        instr->shape().dimensions(output_spatial0_dim);
    const int64_t out_spatial1 =
        instr->shape().dimensions(output_spatial1_dim);

    const WindowDimension& window0 = instr->window().dimensions(0);
    const WindowDimension& window1 = instr->window().dimensions(1);
    auto effective_size = [](int64_t size, int64_t dilation) {
      return size == 0 ? 0 : (size - 1) * dilation + 1;
    };
    for (const WindowDimension* dim : {&window0, &window1}) {
      if (dim->stride() <= 0 || dim->base_dilation() <= 0 ||
          dim->window_dilation() <= 0 || dim->window_reversal()) {
        return absl::UnimplementedError(
            "Metal direct AIR convolution does not support invalid dilation, "
            "stride, or window reversal.");
      }
    }

    const int64_t effective_input0 =
        effective_size(in_spatial0, window0.base_dilation());
    const int64_t effective_input1 =
        effective_size(in_spatial1, window1.base_dilation());
    const int64_t effective_kernel0 =
        effective_size(kernel_spatial0, window0.window_dilation());
    const int64_t effective_kernel1 =
        effective_size(kernel_spatial1, window1.window_dilation());
    const int64_t expected_out_spatial0 =
        (effective_input0 + window0.padding_low() + window0.padding_high() -
         effective_kernel0) /
            window0.stride() +
        1;
    const int64_t expected_out_spatial1 =
        (effective_input1 + window1.padding_low() + window1.padding_high() -
         effective_kernel1) /
            window1.stride() +
        1;
    if (input_batch != output_batch * batch_group_count ||
        in_features != rhs_input_features * feature_group_count ||
        out_features % feature_group_count != 0 ||
        out_features % batch_group_count != 0 ||
        instr->shape().dimensions(output_feature_dim) != out_features ||
        out_spatial0 != expected_out_spatial0 ||
        out_spatial1 != expected_out_spatial1 ||
        window0.size() != kernel_spatial0 ||
        window1.size() != kernel_spatial1) {
      return absl::UnimplementedError(
          "Metal direct AIR 2D convolution dimensions do not match.");
    }

    if (in_features == 0 || rhs_input_features == 0 || in_spatial0 == 0 ||
        in_spatial1 == 0 || kernel_spatial0 == 0 || kernel_spatial1 == 0) {
      return DefaultIrValue(result_type);
    }

    auto emit_coords = [&](const Shape& shape, absl::string_view index,
                           absl::string_view prefix) {
      std::vector<std::string> coords(4);
      std::string remaining(index);
      for (int64_t dim = 0; dim < 4; ++dim) {
        if (dim == 3) {
          coords[dim] = remaining;
          break;
        }
        int64_t stride = 1;
        for (int64_t minor_dim = dim + 1; minor_dim < 4; ++minor_dim) {
          stride *= shape.dimensions(minor_dim);
        }
        std::string coord = NewName(absl::StrCat(prefix, "_coord"));
        std::string rem = NewName(absl::StrCat(prefix, "_rem"));
        body->push_back(absl::StrFormat("  %s = udiv i64 %s, %d", coord,
                                        remaining, stride));
        body->push_back(absl::StrFormat("  %s = urem i64 %s, %d", rem,
                                        remaining, stride));
        coords[dim] = coord;
        remaining = rem;
      }
      return coords;
    };
    auto emit_linear_index = [&](const Shape& shape,
                                 const std::vector<std::string>& coords,
                                 absl::string_view prefix) {
      std::string linear = "0";
      for (int64_t dim = 0; dim < 4; ++dim) {
        if (coords[dim] == "0") {
          continue;
        }
        int64_t stride = 1;
        for (int64_t minor_dim = dim + 1; minor_dim < 4; ++minor_dim) {
          stride *= shape.dimensions(minor_dim);
        }
        std::string term = coords[dim];
        if (stride != 1) {
          term = NewName(absl::StrCat(prefix, "_term"));
          body->push_back(absl::StrFormat("  %s = mul i64 %s, %d", term,
                                          coords[dim], stride));
        }
        if (linear == "0") {
          linear = term;
        } else {
          std::string sum = NewName(absl::StrCat(prefix, "_index"));
          body->push_back(
              absl::StrFormat("  %s = add i64 %s, %s", sum, linear, term));
          linear = sum;
        }
      }
      return linear;
    };
    auto scaled_spatial = [&](absl::string_view output_coord,
                              const WindowDimension& window,
                              int64_t kernel_coord,
                              absl::string_view prefix) {
      std::string coord(output_coord);
      if (window.stride() != 1) {
        std::string scaled = NewName(absl::StrCat(prefix, "_scaled"));
        body->push_back(absl::StrFormat("  %s = mul i64 %s, %d", scaled,
                                        coord, window.stride()));
        coord = scaled;
      }
      const int64_t kernel_offset = kernel_coord * window.window_dilation();
      if (kernel_offset != 0) {
        std::string shifted = NewName(absl::StrCat(prefix, "_kernel"));
        body->push_back(absl::StrFormat("  %s = add i64 %s, %d", shifted,
                                        coord, kernel_offset));
        coord = shifted;
      }
      if (window.padding_low() != 0) {
        std::string padded = NewName(absl::StrCat(prefix, "_padded"));
        body->push_back(absl::StrFormat("  %s = sub i64 %s, %d", padded,
                                        coord, window.padding_low()));
        coord = padded;
      }
      return coord;
    };
    auto emit_undilated_input = [&](absl::string_view dilated_coord,
                                    int64_t effective_input,
                                    int64_t base_dilation,
                                    absl::string_view prefix) {
      std::string ge0 = NewName(absl::StrCat(prefix, "_ge0"));
      std::string lt = NewName(absl::StrCat(prefix, "_lt"));
      std::string in_range = NewName(absl::StrCat(prefix, "_in_range"));
      std::string rem = NewName(absl::StrCat(prefix, "_rem"));
      std::string divisible = NewName(absl::StrCat(prefix, "_divisible"));
      std::string ok = NewName(absl::StrCat(prefix, "_ok"));
      std::string undilated = NewName(absl::StrCat(prefix, "_undilated"));
      std::string safe = NewName(absl::StrCat(prefix, "_safe"));
      body->push_back(absl::StrFormat("  %s = icmp sge i64 %s, 0", ge0,
                                      dilated_coord));
      body->push_back(absl::StrFormat("  %s = icmp slt i64 %s, %d", lt,
                                      dilated_coord, effective_input));
      body->push_back(
          absl::StrFormat("  %s = and i1 %s, %s", in_range, ge0, lt));
      body->push_back(absl::StrFormat("  %s = srem i64 %s, %d", rem,
                                      dilated_coord, base_dilation));
      body->push_back(
          absl::StrFormat("  %s = icmp eq i64 %s, 0", divisible, rem));
      body->push_back(absl::StrFormat("  %s = and i1 %s, %s", ok, in_range,
                                      divisible));
      body->push_back(absl::StrFormat("  %s = sdiv i64 %s, %d", undilated,
                                      dilated_coord, base_dilation));
      body->push_back(absl::StrFormat(
          "  %s = select i1 %s, i64 %s, i64 0", safe, ok, undilated));
      return std::pair<std::string, std::string>(safe, ok);
    };

    std::vector<std::string> out_coords =
        emit_coords(instr->shape(), "%idx", "conv_out");
    const std::string& out_batch = out_coords[output_batch_dim];
    const std::string& oc = out_coords[output_feature_dim];
    const std::string& out0 = out_coords[output_spatial0_dim];
    const std::string& out1 = out_coords[output_spatial1_dim];

    std::string input_batch_coord = out_batch;
    if (batch_group_count != 1) {
      const int64_t out_features_per_batch_group =
          out_features / batch_group_count;
      std::string batch_group = NewName("conv_batch_group");
      std::string batch_group_offset = NewName("conv_batch_group_offset");
      input_batch_coord = NewName("conv_input_batch");
      body->push_back(absl::StrFormat("  %s = udiv i64 %s, %d", batch_group,
                                      oc, out_features_per_batch_group));
      body->push_back(absl::StrFormat("  %s = mul i64 %s, %d",
                                      batch_group_offset, batch_group,
                                      output_batch));
      body->push_back(absl::StrFormat("  %s = add i64 %s, %s",
                                      input_batch_coord, out_batch,
                                      batch_group_offset));
    }
    std::string feature_group = "0";
    if (feature_group_count != 1) {
      const int64_t out_features_per_feature_group =
          out_features / feature_group_count;
      feature_group = NewName("conv_feature_group");
      body->push_back(absl::StrFormat("  %s = udiv i64 %s, %d",
                                      feature_group, oc,
                                      out_features_per_feature_group));
    }

    const bool float_accumulator =
        IsFloatAccumulatorElementType(result_type);
    const bool pred_accumulator = result_type == PRED;
    const char* accumulator_type =
        float_accumulator ? "float" : ValueIrType(result_type);
    std::string accumulator = DefaultIrValue(result_type);
    for (int64_t k0 = 0; k0 < kernel_spatial0; ++k0) {
      for (int64_t k1 = 0; k1 < kernel_spatial1; ++k1) {
        std::string dilated0 =
            scaled_spatial(out0, window0, k0, "conv_input0");
        std::string dilated1 =
            scaled_spatial(out1, window1, k1, "conv_input1");
        auto [safe0, ok0] = emit_undilated_input(
            dilated0, effective_input0, window0.base_dilation(),
            "conv_input0");
        auto [safe1, ok1] = emit_undilated_input(
            dilated1, effective_input1, window1.base_dilation(),
            "conv_input1");
        std::string in_bounds = NewName("conv_in_bounds");
        body->push_back(
            absl::StrFormat("  %s = and i1 %s, %s", in_bounds, ok0, ok1));

        for (int64_t ic = 0; ic < rhs_input_features; ++ic) {
          std::string lhs_feature = absl::StrCat(ic);
          if (feature_group_count != 1) {
            std::string feature_group_offset =
                NewName("conv_feature_group_offset");
            lhs_feature = NewName("conv_lhs_feature");
            body->push_back(absl::StrFormat("  %s = mul i64 %s, %d",
                                            feature_group_offset,
                                            feature_group,
                                            rhs_input_features));
            body->push_back(absl::StrFormat("  %s = add i64 %s, %d",
                                            lhs_feature, feature_group_offset,
                                            ic));
          }
          std::vector<std::string> lhs_coords(4, "0");
          lhs_coords[input_batch_dim] = input_batch_coord;
          lhs_coords[input_feature_dim] = lhs_feature;
          lhs_coords[input_spatial0_dim] = safe0;
          lhs_coords[input_spatial1_dim] = safe1;
          std::string lhs_index =
              emit_linear_index(lhs->shape(), lhs_coords, "conv_lhs");

          std::vector<std::string> rhs_coords(4, "0");
          rhs_coords[kernel_output_feature_dim] = oc;
          rhs_coords[kernel_input_feature_dim] = absl::StrCat(ic);
          rhs_coords[kernel_spatial0_dim] = absl::StrCat(k0);
          rhs_coords[kernel_spatial1_dim] = absl::StrCat(k1);
          std::string rhs_index =
              emit_linear_index(rhs->shape(), rhs_coords, "conv_rhs");
          TF_ASSIGN_OR_RETURN(std::string lhs_value,
                              EmitLoadFromLinearIndex(lhs, lhs_index, body));
          TF_ASSIGN_OR_RETURN(std::string rhs_value,
                              EmitLoadFromLinearIndex(rhs, rhs_index, body));
          TF_ASSIGN_OR_RETURN(lhs_value, EmitConvolutionCastValue(
                                             lhs_value, input_type, result_type,
                                             body));
          TF_ASSIGN_OR_RETURN(rhs_value, EmitConvolutionCastValue(
                                             rhs_value, input_type, result_type,
                                             body));
          std::string product;
          std::string contribution = NewName("conv_contribution");
          if (float_accumulator) {
            product = EmitOp("fmul fast float", lhs_value, rhs_value, body);
            body->push_back(absl::StrFormat(
                "  %s = select i1 %s, float %s, float 0x0000000000000000",
                contribution, in_bounds, product));
            TF_ASSIGN_OR_RETURN(
                accumulator,
                EmitF32ReducerOp(HloOpcode::kAdd, accumulator, contribution,
                                 body));
          } else if (pred_accumulator) {
            product = EmitOp("and i1", lhs_value, rhs_value, body);
            body->push_back(absl::StrFormat(
                "  %s = select i1 %s, i1 %s, i1 false", contribution,
                in_bounds, product));
            accumulator = EmitOp("or i1", accumulator, contribution, body);
          } else {
            product = EmitOp(absl::StrCat("mul ", accumulator_type), lhs_value,
                             rhs_value, body);
            body->push_back(absl::StrFormat(
                "  %s = select i1 %s, %s %s, %s 0", contribution, in_bounds,
                accumulator_type, product, accumulator_type));
            accumulator = EmitOp(absl::StrCat("add ", accumulator_type),
                                 accumulator, contribution, body);
          }
        }
      }
    }
    return accumulator;
  }

  absl::StatusOr<std::string> EmitConvolutionND(
      const HloInstruction* instr, int64_t spatial_rank,
      std::vector<std::string>* body) {
    const HloInstruction* lhs = instr->operand(0);
    const HloInstruction* rhs = instr->operand(1);
    const ConvolutionDimensionNumbers& dnums =
        instr->convolution_dimension_numbers();
    const PrimitiveType conv_type = instr->shape().element_type();
    const int64_t rank = spatial_rank + 2;

    auto is_conv_type = [](PrimitiveType type) {
      switch (type) {
        case F32:
        case F16:
        case BF16:
        case S8:
        case S16:
        case S32:
        case U8:
        case U16:
        case U32:
        case PRED:
          return true;
        default:
          return false;
      }
    };
    if (!ShapeUtil::Equal(instr->shape(), result_shape_) ||
        !is_conv_type(conv_type) || lhs->shape().element_type() != conv_type ||
        rhs->shape().element_type() != conv_type ||
        instr->shape().dimensions().size() != rank ||
        lhs->shape().dimensions().size() != rank ||
        rhs->shape().dimensions().size() != rank ||
        dnums.input_spatial_dimensions_size() != spatial_rank ||
        dnums.kernel_spatial_dimensions_size() != spatial_rank ||
        dnums.output_spatial_dimensions_size() != spatial_rank ||
        instr->window().dimensions().size() != spatial_rank ||
        instr->feature_group_count() <= 0 || instr->batch_group_count() <= 0) {
      return absl::UnimplementedError(
          "Metal direct AIR convolution currently supports only scalar "
          "rank-N convolutions with positive group counts.");
    }

    auto valid_permutation = [&](std::vector<int64_t> dims) {
      if (dims.size() != rank) {
        return false;
      }
      absl::c_sort(dims);
      for (int64_t i = 0; i < rank; ++i) {
        if (dims[i] != i) {
          return false;
        }
      }
      return true;
    };
    std::vector<int64_t> input_dims = {dnums.input_batch_dimension(),
                                       dnums.input_feature_dimension()};
    std::vector<int64_t> kernel_dims = {
        dnums.kernel_output_feature_dimension(),
        dnums.kernel_input_feature_dimension()};
    std::vector<int64_t> output_dims = {dnums.output_batch_dimension(),
                                        dnums.output_feature_dimension()};
    std::vector<int64_t> input_spatial_dims;
    std::vector<int64_t> kernel_spatial_dims;
    std::vector<int64_t> output_spatial_dims;
    input_spatial_dims.reserve(spatial_rank);
    kernel_spatial_dims.reserve(spatial_rank);
    output_spatial_dims.reserve(spatial_rank);
    for (int64_t i = 0; i < spatial_rank; ++i) {
      input_spatial_dims.push_back(dnums.input_spatial_dimensions(i));
      kernel_spatial_dims.push_back(dnums.kernel_spatial_dimensions(i));
      output_spatial_dims.push_back(dnums.output_spatial_dimensions(i));
      input_dims.push_back(input_spatial_dims.back());
      kernel_dims.push_back(kernel_spatial_dims.back());
      output_dims.push_back(output_spatial_dims.back());
    }
    if (!valid_permutation(input_dims) || !valid_permutation(kernel_dims) ||
        !valid_permutation(output_dims)) {
      return absl::UnimplementedError(
          "Metal direct AIR convolution dimension numbers must be valid "
          "rank-N permutations.");
    }

    const int64_t input_batch_dim = dnums.input_batch_dimension();
    const int64_t input_feature_dim = dnums.input_feature_dimension();
    const int64_t kernel_output_feature_dim =
        dnums.kernel_output_feature_dimension();
    const int64_t kernel_input_feature_dim =
        dnums.kernel_input_feature_dimension();
    const int64_t output_batch_dim = dnums.output_batch_dimension();
    const int64_t output_feature_dim = dnums.output_feature_dimension();
    const int64_t feature_group_count = instr->feature_group_count();
    const int64_t batch_group_count = instr->batch_group_count();
    const int64_t input_batch = lhs->shape().dimensions(input_batch_dim);
    const int64_t output_batch =
        instr->shape().dimensions(output_batch_dim);
    const int64_t in_features = lhs->shape().dimensions(input_feature_dim);
    const int64_t out_features =
        rhs->shape().dimensions(kernel_output_feature_dim);
    const int64_t rhs_input_features =
        rhs->shape().dimensions(kernel_input_feature_dim);

    auto effective_size = [](int64_t size, int64_t dilation) {
      return size == 0 ? 0 : (size - 1) * dilation + 1;
    };
    std::vector<int64_t> input_spatial_sizes;
    std::vector<int64_t> kernel_spatial_sizes;
    std::vector<int64_t> output_spatial_sizes;
    std::vector<int64_t> effective_input_sizes;
    input_spatial_sizes.reserve(spatial_rank);
    kernel_spatial_sizes.reserve(spatial_rank);
    output_spatial_sizes.reserve(spatial_rank);
    effective_input_sizes.reserve(spatial_rank);
    for (int64_t i = 0; i < spatial_rank; ++i) {
      const WindowDimension& window = instr->window().dimensions(i);
      if (window.stride() <= 0 || window.base_dilation() <= 0 ||
          window.window_dilation() <= 0 || window.window_reversal()) {
        return absl::UnimplementedError(
            "Metal direct AIR convolution does not support invalid dilation, "
            "stride, or window reversal.");
      }
      const int64_t input_size =
          lhs->shape().dimensions(input_spatial_dims[i]);
      const int64_t kernel_size =
          rhs->shape().dimensions(kernel_spatial_dims[i]);
      const int64_t output_size =
          instr->shape().dimensions(output_spatial_dims[i]);
      const int64_t effective_input =
          effective_size(input_size, window.base_dilation());
      const int64_t effective_kernel =
          effective_size(kernel_size, window.window_dilation());
      const int64_t expected_output =
          (effective_input + window.padding_low() + window.padding_high() -
           effective_kernel) /
              window.stride() +
          1;
      if (output_size != expected_output || window.size() != kernel_size) {
        return absl::UnimplementedError(
            "Metal direct AIR convolution spatial dimensions do not match.");
      }
      input_spatial_sizes.push_back(input_size);
      kernel_spatial_sizes.push_back(kernel_size);
      output_spatial_sizes.push_back(output_size);
      effective_input_sizes.push_back(effective_input);
    }
    if (input_batch != output_batch * batch_group_count ||
        in_features != rhs_input_features * feature_group_count ||
        out_features % feature_group_count != 0 ||
        out_features % batch_group_count != 0 ||
        instr->shape().dimensions(output_feature_dim) != out_features) {
      return absl::UnimplementedError(
          "Metal direct AIR convolution dimensions do not match.");
    }
    if (in_features == 0 || rhs_input_features == 0) {
      return DefaultIrValue(conv_type);
    }
    for (int64_t i = 0; i < spatial_rank; ++i) {
      if (input_spatial_sizes[i] == 0 || kernel_spatial_sizes[i] == 0) {
        return DefaultIrValue(conv_type);
      }
    }

    auto emit_coords = [&](const Shape& shape, absl::string_view index,
                           absl::string_view prefix) {
      std::vector<std::string> coords(rank);
      std::string remaining(index);
      for (int64_t dim = 0; dim < rank; ++dim) {
        if (dim == rank - 1) {
          coords[dim] = remaining;
          break;
        }
        int64_t stride = 1;
        for (int64_t minor_dim = dim + 1; minor_dim < rank; ++minor_dim) {
          stride *= shape.dimensions(minor_dim);
        }
        std::string coord = NewName(absl::StrCat(prefix, "_coord"));
        std::string rem = NewName(absl::StrCat(prefix, "_rem"));
        body->push_back(absl::StrFormat("  %s = udiv i64 %s, %d", coord,
                                        remaining, stride));
        body->push_back(absl::StrFormat("  %s = urem i64 %s, %d", rem,
                                        remaining, stride));
        coords[dim] = coord;
        remaining = rem;
      }
      return coords;
    };
    auto emit_linear_index = [&](const Shape& shape,
                                 const std::vector<std::string>& coords,
                                 absl::string_view prefix) {
      std::string linear = "0";
      for (int64_t dim = 0; dim < rank; ++dim) {
        if (coords[dim] == "0") {
          continue;
        }
        int64_t stride = 1;
        for (int64_t minor_dim = dim + 1; minor_dim < rank; ++minor_dim) {
          stride *= shape.dimensions(minor_dim);
        }
        std::string term = coords[dim];
        if (stride != 1) {
          term = NewName(absl::StrCat(prefix, "_term"));
          body->push_back(absl::StrFormat("  %s = mul i64 %s, %d", term,
                                          coords[dim], stride));
        }
        if (linear == "0") {
          linear = term;
        } else {
          std::string sum = NewName(absl::StrCat(prefix, "_index"));
          body->push_back(
              absl::StrFormat("  %s = add i64 %s, %s", sum, linear, term));
          linear = sum;
        }
      }
      return linear;
    };
    auto scaled_spatial = [&](absl::string_view output_coord,
                              const WindowDimension& window,
                              int64_t kernel_coord,
                              absl::string_view prefix) {
      std::string coord(output_coord);
      if (window.stride() != 1) {
        std::string scaled = NewName(absl::StrCat(prefix, "_scaled"));
        body->push_back(absl::StrFormat("  %s = mul i64 %s, %d", scaled,
                                        coord, window.stride()));
        coord = scaled;
      }
      const int64_t kernel_offset = kernel_coord * window.window_dilation();
      if (kernel_offset != 0) {
        std::string shifted = NewName(absl::StrCat(prefix, "_kernel"));
        body->push_back(absl::StrFormat("  %s = add i64 %s, %d", shifted,
                                        coord, kernel_offset));
        coord = shifted;
      }
      if (window.padding_low() != 0) {
        std::string padded = NewName(absl::StrCat(prefix, "_padded"));
        body->push_back(absl::StrFormat("  %s = sub i64 %s, %d", padded,
                                        coord, window.padding_low()));
        coord = padded;
      }
      return coord;
    };
    auto emit_undilated_input = [&](absl::string_view dilated_coord,
                                    int64_t effective_input,
                                    int64_t base_dilation,
                                    absl::string_view prefix) {
      std::string ge0 = NewName(absl::StrCat(prefix, "_ge0"));
      std::string lt = NewName(absl::StrCat(prefix, "_lt"));
      std::string in_range = NewName(absl::StrCat(prefix, "_in_range"));
      std::string rem = NewName(absl::StrCat(prefix, "_rem"));
      std::string divisible = NewName(absl::StrCat(prefix, "_divisible"));
      std::string ok = NewName(absl::StrCat(prefix, "_ok"));
      std::string undilated = NewName(absl::StrCat(prefix, "_undilated"));
      std::string safe = NewName(absl::StrCat(prefix, "_safe"));
      body->push_back(absl::StrFormat("  %s = icmp sge i64 %s, 0", ge0,
                                      dilated_coord));
      body->push_back(absl::StrFormat("  %s = icmp slt i64 %s, %d", lt,
                                      dilated_coord, effective_input));
      body->push_back(
          absl::StrFormat("  %s = and i1 %s, %s", in_range, ge0, lt));
      body->push_back(absl::StrFormat("  %s = srem i64 %s, %d", rem,
                                      dilated_coord, base_dilation));
      body->push_back(
          absl::StrFormat("  %s = icmp eq i64 %s, 0", divisible, rem));
      body->push_back(absl::StrFormat("  %s = and i1 %s, %s", ok, in_range,
                                      divisible));
      body->push_back(absl::StrFormat("  %s = sdiv i64 %s, %d", undilated,
                                      dilated_coord, base_dilation));
      body->push_back(absl::StrFormat(
          "  %s = select i1 %s, i64 %s, i64 0", safe, ok, undilated));
      return std::pair<std::string, std::string>(safe, ok);
    };

    std::vector<std::string> out_coords =
        emit_coords(instr->shape(), "%idx", "conv_nd_out");
    const std::string& out_batch = out_coords[output_batch_dim];
    const std::string& oc = out_coords[output_feature_dim];

    std::string input_batch_coord = out_batch;
    if (batch_group_count != 1) {
      const int64_t out_features_per_batch_group =
          out_features / batch_group_count;
      std::string batch_group = NewName("conv_nd_batch_group");
      std::string batch_group_offset =
          NewName("conv_nd_batch_group_offset");
      input_batch_coord = NewName("conv_nd_input_batch");
      body->push_back(absl::StrFormat("  %s = udiv i64 %s, %d", batch_group,
                                      oc, out_features_per_batch_group));
      body->push_back(absl::StrFormat("  %s = mul i64 %s, %d",
                                      batch_group_offset, batch_group,
                                      output_batch));
      body->push_back(absl::StrFormat("  %s = add i64 %s, %s",
                                      input_batch_coord, out_batch,
                                      batch_group_offset));
    }
    std::string feature_group = "0";
    if (feature_group_count != 1) {
      const int64_t out_features_per_feature_group =
          out_features / feature_group_count;
      feature_group = NewName("conv_nd_feature_group");
      body->push_back(absl::StrFormat("  %s = udiv i64 %s, %d",
                                      feature_group, oc,
                                      out_features_per_feature_group));
    }

    const bool float_accumulator =
        conv_type == F32 || conv_type == F16 || conv_type == BF16;
    const bool pred_accumulator = conv_type == PRED;
    const char* accumulator_type =
        float_accumulator ? "float" : ValueIrType(conv_type);
    std::string accumulator = DefaultIrValue(conv_type);
    std::vector<int64_t> kernel_coords(spatial_rank, 0);
    std::function<absl::Status(int64_t)> emit_kernel_loop;
    emit_kernel_loop = [&](int64_t spatial_dim) -> absl::Status {
      if (spatial_dim != spatial_rank) {
        for (int64_t k = 0; k < kernel_spatial_sizes[spatial_dim]; ++k) {
          kernel_coords[spatial_dim] = k;
          TF_RETURN_IF_ERROR(emit_kernel_loop(spatial_dim + 1));
        }
        return absl::OkStatus();
      }

      std::vector<std::string> safe_spatial(spatial_rank);
      std::string in_bounds = "true";
      for (int64_t i = 0; i < spatial_rank; ++i) {
        const WindowDimension& window = instr->window().dimensions(i);
        std::string dilated =
            scaled_spatial(out_coords[output_spatial_dims[i]], window,
                           kernel_coords[i], "conv_nd_input");
        auto [safe, ok] = emit_undilated_input(
            dilated, effective_input_sizes[i], window.base_dilation(),
            "conv_nd_input");
        safe_spatial[i] = safe;
        if (in_bounds == "true") {
          in_bounds = ok;
        } else {
          std::string combined = NewName("conv_nd_in_bounds");
          body->push_back(absl::StrFormat("  %s = and i1 %s, %s", combined,
                                          in_bounds, ok));
          in_bounds = combined;
        }
      }

      for (int64_t ic = 0; ic < rhs_input_features; ++ic) {
        std::string lhs_feature = absl::StrCat(ic);
        if (feature_group_count != 1) {
          std::string feature_group_offset =
              NewName("conv_nd_feature_group_offset");
          lhs_feature = NewName("conv_nd_lhs_feature");
          body->push_back(absl::StrFormat("  %s = mul i64 %s, %d",
                                          feature_group_offset, feature_group,
                                          rhs_input_features));
          body->push_back(absl::StrFormat("  %s = add i64 %s, %d",
                                          lhs_feature, feature_group_offset,
                                          ic));
        }
        std::vector<std::string> lhs_coords(rank, "0");
        lhs_coords[input_batch_dim] = input_batch_coord;
        lhs_coords[input_feature_dim] = lhs_feature;
        for (int64_t i = 0; i < spatial_rank; ++i) {
          lhs_coords[input_spatial_dims[i]] = safe_spatial[i];
        }
        std::string lhs_index =
            emit_linear_index(lhs->shape(), lhs_coords, "conv_nd_lhs");

        std::vector<std::string> rhs_coords(rank, "0");
        rhs_coords[kernel_output_feature_dim] = oc;
        rhs_coords[kernel_input_feature_dim] = absl::StrCat(ic);
        for (int64_t i = 0; i < spatial_rank; ++i) {
          rhs_coords[kernel_spatial_dims[i]] = absl::StrCat(kernel_coords[i]);
        }
        std::string rhs_index =
            emit_linear_index(rhs->shape(), rhs_coords, "conv_nd_rhs");
        TF_ASSIGN_OR_RETURN(std::string lhs_value,
                            EmitLoadFromLinearIndex(lhs, lhs_index, body));
        TF_ASSIGN_OR_RETURN(std::string rhs_value,
                            EmitLoadFromLinearIndex(rhs, rhs_index, body));
        std::string product;
        std::string contribution = NewName("conv_nd_contribution");
        if (float_accumulator) {
          product = EmitOp("fmul fast float", lhs_value, rhs_value, body);
          body->push_back(absl::StrFormat(
              "  %s = select i1 %s, float %s, float 0x0000000000000000",
              contribution, in_bounds, product));
          TF_ASSIGN_OR_RETURN(
              accumulator,
              EmitF32ReducerOp(HloOpcode::kAdd, accumulator, contribution,
                               body));
        } else if (pred_accumulator) {
          product = EmitOp("and i1", lhs_value, rhs_value, body);
          body->push_back(absl::StrFormat(
              "  %s = select i1 %s, i1 %s, i1 false", contribution,
              in_bounds, product));
          accumulator = EmitOp("or i1", accumulator, contribution, body);
        } else {
          product = EmitOp(absl::StrCat("mul ", accumulator_type), lhs_value,
                           rhs_value, body);
          body->push_back(absl::StrFormat(
              "  %s = select i1 %s, %s %s, %s 0", contribution, in_bounds,
              accumulator_type, product, accumulator_type));
          accumulator = EmitOp(absl::StrCat("add ", accumulator_type),
                               accumulator, contribution, body);
        }
      }
      return absl::OkStatus();
    };
    TF_RETURN_IF_ERROR(emit_kernel_loop(0));
    return accumulator;
  }

  absl::StatusOr<std::string> EmitConditional(
      const HloInstruction* instr, std::vector<std::string>* body) {
    if (!IsSupportedElementwiseArray(instr->shape()) ||
        !ShapeUtil::Equal(instr->shape(), result_shape_) ||
        instr->branch_count() != 2 || instr->operand_count() != 3) {
      return absl::UnimplementedError(
          "Metal direct AIR conditional currently supports only two-way "
          "conditionals with array results matching the final result.");
    }

    if (instr->operand(0)->shape().element_type() == PRED) {
      TF_ASSIGN_OR_RETURN(std::string pred,
                          EmitValue(instr->operand(0), /*force_scalar=*/true,
                                    body));
      TF_ASSIGN_OR_RETURN(std::string on_true,
                          EmitConditionalBranch(instr->true_computation(),
                                                instr->operand(1), body));
      TF_ASSIGN_OR_RETURN(std::string on_false,
                          EmitConditionalBranch(instr->false_computation(),
                                                instr->operand(2), body));
      return EmitTypedSelect(instr->shape().element_type(), pred, on_true,
                             on_false, body, "conditional_select");
    }

    if (instr->operand(0)->shape().element_type() != S32) {
      return absl::UnimplementedError(
          "Metal direct AIR conditional branch index must be pred or s32.");
    }
    TF_ASSIGN_OR_RETURN(std::string branch_index,
                        EmitValue(instr->operand(0), /*force_scalar=*/true,
                                  body));
    TF_ASSIGN_OR_RETURN(std::string branch0,
                        EmitConditionalBranch(instr->branch_computation(0),
                                              instr->operand(1), body));
    TF_ASSIGN_OR_RETURN(std::string branch1,
                        EmitConditionalBranch(instr->branch_computation(1),
                                              instr->operand(2), body));
    std::string is_branch1 = NewName("conditional_is_branch1");
    body->push_back(absl::StrFormat("  %s = icmp eq i32 %s, 1", is_branch1,
                                    branch_index));
    return EmitTypedSelect(instr->shape().element_type(), is_branch1, branch1,
                           branch0, body, "conditional_select");
  }

  absl::StatusOr<std::string> EmitConditionalBranch(
      const HloComputation* branch, const HloInstruction* argument,
      std::vector<std::string>* body) {
    if (branch->num_parameters() != 1) {
      return absl::UnimplementedError(
          "Metal direct AIR conditional branches must take one parameter.");
    }
    CallParameterScope scope;
    scope.computation = branch;
    scope.arguments[0] = argument;
    call_parameter_scopes_.push_back(std::move(scope));
    absl::Cleanup pop_scope = [this] { call_parameter_scopes_.pop_back(); };
    return EmitValue(branch->root_instruction(), /*force_scalar=*/false, body);
  }

  absl::StatusOr<std::string> EmitConcatenateAtIndex(
      const HloInstruction* instr, absl::string_view index,
      absl::string_view name_prefix, std::vector<std::string>* body) {
    if (!IsSupportedElementwiseArray(instr->shape()) ||
        instr->dimensions().size() != 1 || instr->operand_count() == 0) {
      return absl::UnimplementedError(
          "Metal direct AIR concatenate requires a supported elementwise "
          "array and one concatenate dimension.");
    }
    const int64_t rank = instr->shape().dimensions().size();
    const int64_t concat_dim = instr->dimensions()[0];
    if ((rank < 1 || rank > 3) || concat_dim < 0 || concat_dim >= rank) {
      return absl::UnimplementedError(
          "Metal direct AIR concatenate currently supports only rank-1, "
          "rank-2, and rank-3 concatenates.");
    }

    std::vector<std::string> coords(rank);
    std::string remaining(index);
    for (int64_t dim = 0; dim < rank; ++dim) {
      if (dim == rank - 1) {
        coords[dim] = remaining;
        break;
      }
      int64_t stride = 1;
      for (int64_t minor_dim = dim + 1; minor_dim < rank; ++minor_dim) {
        stride *= instr->shape().dimensions(minor_dim);
      }
      std::string coord = NewName(absl::StrCat(name_prefix, "_coord"));
      std::string rem = NewName(absl::StrCat(name_prefix, "_rem"));
      body->push_back(absl::StrFormat("  %s = udiv i64 %s, %d", coord,
                                      remaining, stride));
      body->push_back(absl::StrFormat("  %s = urem i64 %s, %d", rem,
                                      remaining, stride));
      coords[dim] = coord;
      remaining = rem;
    }

    std::string selected;
    int64_t output_offset = 0;
    for (const HloInstruction* operand : instr->operands()) {
      if (operand->shape().element_type() != instr->shape().element_type() ||
          operand->shape().dimensions().size() != rank) {
        return absl::UnimplementedError(
            "Metal direct AIR concatenate operand rank and element type must "
            "match the result.");
      }
      for (int64_t dim = 0; dim < rank; ++dim) {
        if (dim != concat_dim &&
            operand->shape().dimensions(dim) != instr->shape().dimensions(dim)) {
          return absl::UnimplementedError(
              "Metal direct AIR concatenate non-concatenated dimensions must "
              "match the result.");
        }
      }

      const int64_t operand_concat_size =
          operand->shape().dimensions(concat_dim);
      if (operand_concat_size == 0) {
        return absl::UnimplementedError(
            "Metal direct AIR concatenate does not support empty operands.");
      }
      const int64_t operand_end = output_offset + operand_concat_size;

      std::string in_range;
      const std::string& coord = coords[concat_dim];
      if (output_offset == 0 &&
          operand_end == instr->shape().dimensions(concat_dim)) {
        in_range = "true";
      } else if (output_offset == 0) {
        in_range = NewName(absl::StrCat(name_prefix, "_in_range"));
        body->push_back(absl::StrFormat("  %s = icmp ult i64 %s, %d",
                                        in_range, coord, operand_end));
      } else if (operand_end == instr->shape().dimensions(concat_dim)) {
        in_range = NewName(absl::StrCat(name_prefix, "_in_range"));
        body->push_back(absl::StrFormat("  %s = icmp uge i64 %s, %d",
                                        in_range, coord, output_offset));
      } else {
        std::string ge_start = NewName(absl::StrCat(name_prefix, "_ge_start"));
        std::string lt_end = NewName(absl::StrCat(name_prefix, "_lt_end"));
        in_range = NewName(absl::StrCat(name_prefix, "_in_range"));
        body->push_back(absl::StrFormat("  %s = icmp uge i64 %s, %d",
                                        ge_start, coord, output_offset));
        body->push_back(absl::StrFormat("  %s = icmp ult i64 %s, %d", lt_end,
                                        coord, operand_end));
        body->push_back(absl::StrFormat("  %s = and i1 %s, %s", in_range,
                                        ge_start, lt_end));
      }

      std::vector<std::string> local_coords = coords;
      if (output_offset != 0) {
        local_coords[concat_dim] =
            NewName(absl::StrCat(name_prefix, "_local_coord"));
        body->push_back(absl::StrFormat("  %s = sub i64 %s, %d",
                                        local_coords[concat_dim], coord,
                                        output_offset));
      }

      std::string local_index = "0";
      for (int64_t dim = 0; dim < rank; ++dim) {
        int64_t stride = 1;
        for (int64_t minor_dim = dim + 1; minor_dim < rank; ++minor_dim) {
          stride *= operand->shape().dimensions(minor_dim);
        }
        std::string term = local_coords[dim];
        if (stride != 1) {
          term = NewName(absl::StrCat(name_prefix, "_local_term"));
          body->push_back(absl::StrFormat("  %s = mul i64 %s, %d", term,
                                          local_coords[dim], stride));
        }
        if (local_index == "0") {
          local_index = term;
        } else {
          std::string sum = NewName(absl::StrCat(name_prefix, "_local_index"));
          body->push_back(absl::StrFormat("  %s = add i64 %s, %s", sum,
                                          local_index, term));
          local_index = sum;
        }
      }

      std::string safe_index = local_index;
      if (in_range != "true") {
        safe_index = NewName(absl::StrCat(name_prefix, "_safe_index"));
        body->push_back(absl::StrFormat(
            "  %s = select i1 %s, i64 %s, i64 0", safe_index, in_range,
            local_index));
      }

      TF_ASSIGN_OR_RETURN(std::string value,
                          EmitLoadFromLinearIndex(operand, safe_index, body));
      if (selected.empty()) {
        selected = value;
      } else {
        selected = EmitTypedSelect(instr->shape().element_type(), in_range,
                                   value, selected, body,
                                   absl::StrCat(name_prefix, "_select"));
      }
      output_offset = operand_end;
    }

    if (output_offset != instr->shape().dimensions(concat_dim)) {
      return absl::InvalidArgumentError(
          "Metal direct AIR concatenate operand sizes do not match result.");
    }
    return selected;
  }

  absl::StatusOr<std::string> EmitConcatenate(const HloInstruction* instr,
                                              std::vector<std::string>* body) {
    if (!ShapeUtil::Equal(instr->shape(), result_shape_)) {
      return absl::UnimplementedError(
          "Metal direct AIR elementwise concatenate must match the final "
          "result shape.");
    }
    return EmitConcatenateAtIndex(instr, "%idx", "concat", body);
  }

  absl::StatusOr<std::string> EmitConcatenateLinearIndex(
      const HloInstruction* instr, absl::string_view index,
      std::vector<std::string>* body) {
    return EmitConcatenateAtIndex(instr, index, "linear_concat", body);
  }

  absl::StatusOr<bool> SortDescending(const HloInstruction* instr) {
    const HloInstruction* root = instr->to_apply()->root_instruction();
    if (root->opcode() == HloOpcode::kCompare &&
        root->operand_count() == 2 &&
        root->operand(0)->opcode() == HloOpcode::kParameter &&
        root->operand(1)->opcode() == HloOpcode::kParameter) {
      if (root->comparison_direction() == ComparisonDirection::kLt) {
        return false;
      }
      if (root->comparison_direction() == ComparisonDirection::kGt) {
        return true;
      }
    }

    // JAX lowers f32 sort to a total-order comparator. Its expanded HLO is
    // intentionally more complex than a direct compare, but the common sort
    // direction is ascending.
    return false;
  }

  absl::StatusOr<std::string> EmitSort(const HloInstruction* instr,
                                       std::vector<std::string>* body) {
    if (IsOrderStatisticArray(instr->shape()) && instr->operand_count() == 1 &&
        instr->operand(0)->shape().element_type() ==
            instr->shape().element_type() &&
        instr->shape().dimensions().size() == 2 &&
        instr->operand(0)->shape().dimensions().size() == 2 &&
        instr->dimensions().size() == 1 && instr->dimensions()[0] == 1 &&
        ShapeUtil::Equal(instr->shape(), instr->operand(0)->shape()) &&
        ShapeUtil::Equal(instr->shape(), result_shape_)) {
      TF_ASSIGN_OR_RETURN(bool descending, SortDescending(instr));
      return EmitRank2OrderStatistic(
          instr->operand(0), instr->shape().dimensions(0),
          instr->shape().dimensions(1), instr->shape().dimensions(1),
          descending, body, /*return_index=*/false);
    }

    if (!IsOrderStatisticArray(instr->shape()) || instr->operand_count() != 1 ||
        instr->operand(0)->shape().element_type() !=
            instr->shape().element_type() ||
        instr->shape().dimensions().size() != 1 ||
        instr->operand(0)->shape().dimensions().size() != 1 ||
        instr->dimensions().size() != 1 || instr->dimensions()[0] != 0 ||
        !ShapeUtil::Equal(instr->shape(), result_shape_)) {
      return absl::UnimplementedError(
          "Metal direct AIR sort currently supports only single-operand "
          "rank-1/rank-2 f32/s32/u16 sorts along the minor dimension.");
    }

    TF_ASSIGN_OR_RETURN(bool descending, SortDescending(instr));
    return EmitRank1OrderStatistic(instr->operand(0),
                                   ShapeUtil::ElementsIn(instr->shape()),
                                   descending, body,
                                   /*return_index=*/false);
  }

  absl::StatusOr<std::string> EmitGetTupleElement(
      const HloInstruction* instr, std::vector<std::string>* body) {
    if (instr->operand(0)->opcode() == HloOpcode::kWhile) {
      return EmitSimpleWhileAccumulator(instr, body);
    }
    if (instr->operand(0)->opcode() == HloOpcode::kReduce) {
      return EmitArgmaxReduceTupleElement(instr, body);
    }
    if (instr->operand(0)->opcode() != HloOpcode::kTopK ||
        !ShapeUtil::Equal(instr->shape(), result_shape_)) {
      return absl::UnimplementedError(
          "Metal direct AIR get-tuple-element currently supports only the "
          "values or indices output of rank-1/rank-2 f32 topk.");
    }

    return EmitTopKElement(Cast<HloTopKInstruction>(instr->operand(0)),
                           instr->tuple_index(), body);
  }

  absl::StatusOr<std::string> EmitTopKElement(
      const HloTopKInstruction* topk, int64_t tuple_index,
      std::vector<std::string>* body) {
    const HloInstruction* input = topk->operand(0);
    const bool supported_value_type =
        IsF32Array(input->shape()) || IsS32Array(input->shape());
    if (supported_value_type && input->shape().dimensions().size() == 2 &&
        result_shape_.dimensions().size() == 2 &&
        result_shape_.dimensions(0) == input->shape().dimensions(0) &&
        result_shape_.dimensions(1) == topk->k()) {
      if (tuple_index == 1 && IsS32Array(result_shape_)) {
        return EmitRank2OrderStatistic(
            input, input->shape().dimensions(0), input->shape().dimensions(1),
            topk->k(), topk->largest(), body, /*return_index=*/true);
      }
      if (tuple_index == 0 &&
          result_shape_.element_type() == input->shape().element_type()) {
        return EmitRank2OrderStatistic(
            input, input->shape().dimensions(0), input->shape().dimensions(1),
            topk->k(), topk->largest(), body, /*return_index=*/false);
      }
      return absl::UnimplementedError(
          "Metal direct AIR rank-2 topk get-tuple-element supports only f32 "
          "or s32 values and s32 indices.");
    }

    if (!supported_value_type || input->shape().dimensions().size() != 1 ||
        result_shape_.dimensions().size() != 1 ||
        ShapeUtil::ElementsIn(result_shape_) != topk->k()) {
      return absl::UnimplementedError(
          "Metal direct AIR topk values currently supports only rank-1 f32 or "
          "s32 inputs with a rank-1 values result.");
    }

    if (tuple_index == 1 && IsS32Array(result_shape_)) {
      return EmitRank1OrderStatistic(
          input, ShapeUtil::ElementsIn(input->shape()), topk->largest(), body,
          /*return_index=*/true);
    }
    if (tuple_index != 0 ||
        result_shape_.element_type() != input->shape().element_type()) {
      return absl::UnimplementedError(
          "Metal direct AIR topk get-tuple-element supports only f32 or s32 "
          "values and s32 indices.");
    }

    return EmitRank1OrderStatistic(
        input, ShapeUtil::ElementsIn(input->shape()), topk->largest(), body,
        /*return_index=*/false);
  }

  absl::StatusOr<std::string> EmitArgmaxReduceTupleElement(
      const HloInstruction* instr, std::vector<std::string>* body) {
    const HloInstruction* reduce = instr->operand(0);
    if (instr->tuple_index() != 1 || !IsScalarLikeS32(instr->shape()) ||
        !ShapeUtil::Equal(instr->shape(), result_shape_) ||
        reduce->operand_count() != 4 || reduce->dimensions().size() != 1 ||
        reduce->dimensions()[0] != 0) {
      return absl::UnimplementedError(
          "Metal direct AIR argmax reduce currently supports only a rank-1 "
          "f32 input reduced to a scalar s32 index.");
    }
    const HloInstruction* input = reduce->operand(0);
    const HloInstruction* iota = reduce->operand(1);
    const HloInstruction* init_value = reduce->operand(2);
    const HloInstruction* init_index = reduce->operand(3);
    if (!IsF32Array(input->shape()) || input->shape().dimensions().size() != 1 ||
        iota->opcode() != HloOpcode::kIota || !IsS32Array(iota->shape()) ||
        iota->shape().dimensions().size() != 1 ||
        iota->shape().dimensions(0) != input->shape().dimensions(0) ||
        !init_value->IsConstant() || !IsScalarLikeF32(init_value->shape()) ||
        !init_index->IsConstant() || !IsScalarLikeS32(init_index->shape())) {
      return absl::UnimplementedError(
          "Metal direct AIR argmax reduce requires rank-1 f32 input, rank-1 "
          "s32 iota, and scalar constant initial values.");
    }

    const int64_t input_elements = input->shape().dimensions(0);
    if (input_elements <= 0) {
      return absl::UnimplementedError(
          "Metal direct AIR argmax reduce does not support empty inputs.");
    }

    const int32_t init_index_value =
        init_index->literal().Get<int32_t>({});
    const float init_value_value = init_value->literal().Get<float>({});
    const std::string header = NewLabel("argmax_header");
    const std::string loop_body = NewLabel("argmax_body");
    const std::string done = NewLabel("argmax_done");
    const std::string i = NewName("argmax_i");
    const std::string best_index = NewName("argmax_best_index");
    const std::string best_value = NewName("argmax_best_value");
    const std::string in_range = NewName("argmax_in_range");
    const std::string i64 = NewName("argmax_i64");
    const std::string better = NewName("argmax_better");
    const std::string next_best_value = NewName("argmax_next_best_value");
    const std::string next_best_index = NewName("argmax_next_best_index");
    const std::string next_i = NewName("argmax_next_i");

    body->push_back(absl::StrFormat("  br label %%%s", header));
    body->push_back(absl::StrFormat("%s:", header));
    body->push_back(absl::StrFormat(
        "  %s = phi i32 [ 0, %%body ], [ %s, %%%s ]", i, next_i,
        loop_body));
    body->push_back(absl::StrFormat(
        "  %s = phi i32 [ %d, %%body ], [ %s, %%%s ]", best_index,
        init_index_value, next_best_index, loop_body));
    body->push_back(absl::StrFormat(
        "  %s = phi float [ %s, %%body ], [ %s, %%%s ]", best_value,
        FloatLiteral(init_value_value), next_best_value, loop_body));
    body->push_back(absl::StrFormat("  %s = icmp ult i32 %s, %d", in_range, i,
                                    input_elements));
    body->push_back(absl::StrFormat("  br i1 %s, label %%%s, label %%%s",
                                    in_range, loop_body, done));

    body->push_back(absl::StrFormat("%s:", loop_body));
    body->push_back(absl::StrFormat("  %s = zext i32 %s to i64", i64, i));
    TF_ASSIGN_OR_RETURN(std::string value,
                        EmitLoadFromLinearIndex(input, i64, body));
    body->push_back(absl::StrFormat("  %s = fcmp fast ogt float %s, %s",
                                    better, value, best_value));
    body->push_back(absl::StrFormat(
        "  %s = select i1 %s, float %s, float %s", next_best_value, better,
        value, best_value));
    body->push_back(absl::StrFormat(
        "  %s = select i1 %s, i32 %s, i32 %s", next_best_index, better, i,
        best_index));
    body->push_back(absl::StrFormat("  %s = add i32 %s, 1", next_i, i));
    body->push_back(absl::StrFormat("  br label %%%s", header));

    body->push_back(absl::StrFormat("%s:", done));
    return best_index;
  }

  absl::StatusOr<std::string> EmitSimpleWhileAccumulator(
      const HloInstruction* instr, std::vector<std::string>* body) {
    if (!ShapeUtil::IsEffectiveScalar(instr->shape()) &&
        instr->tuple_index() == 1 && IsF32Array(instr->shape()) &&
        ShapeUtil::Equal(instr->shape(), result_shape_)) {
      return EmitSimpleWhileVectorAccumulator(instr, body);
    }

    const HloInstruction* while_instr = instr->operand(0);
    if (instr->tuple_index() != 1 || !IsScalarLikeF32(instr->shape()) ||
        !ShapeUtil::Equal(instr->shape(), result_shape_) ||
        while_instr->operand_count() != 1 ||
        while_instr->operand(0)->opcode() != HloOpcode::kTuple ||
        while_instr->operand(0)->operand_count() != 3) {
      return absl::UnimplementedError(
          "Metal direct AIR while currently supports only returning the f32 "
          "accumulator from a narrow scalar loop tuple.");
    }

    const HloInstruction* init_tuple = while_instr->operand(0);
    const HloInstruction* init_iter = init_tuple->operand(0);
    const HloInstruction* init_acc = init_tuple->operand(1);
    const HloInstruction* input = init_tuple->operand(2);
    if (!init_iter->IsConstant() || !IsScalarLikeS32(init_iter->shape()) ||
        !init_acc->IsConstant() || !IsScalarLikeF32(init_acc->shape()) ||
        !IsF32Array(input->shape()) || input->shape().dimensions().size() != 1) {
      return absl::UnimplementedError(
          "Metal direct AIR while accumulator requires constant scalar init "
          "values and a rank-1 f32 input.");
    }

    const HloInstruction* cond_root =
        while_instr->while_condition()->root_instruction();
    if (cond_root->opcode() != HloOpcode::kCompare ||
        cond_root->comparison_direction() != ComparisonDirection::kLt ||
        cond_root->operand(1)->opcode() != HloOpcode::kConstant ||
        !IsScalarLikeS32(cond_root->operand(1)->shape())) {
      return absl::UnimplementedError(
          "Metal direct AIR while accumulator requires an i < constant "
          "condition.");
    }
    const int32_t limit = cond_root->operand(1)->literal().Get<int32_t>({});

    const HloInstruction* body_root =
        while_instr->while_body()->root_instruction();
    if (body_root->opcode() != HloOpcode::kTuple ||
        body_root->operand_count() != 3 ||
        body_root->operand(0)->opcode() != HloOpcode::kAdd ||
        body_root->operand(1)->opcode() != HloOpcode::kAdd) {
      return absl::UnimplementedError(
          "Metal direct AIR while accumulator requires tuple(i + c, acc + "
          "value, input) body.");
    }
    TF_ASSIGN_OR_RETURN(int32_t step, ExtractS32ConstantAddend(body_root->operand(0)));
    if (step <= 0) {
      return absl::UnimplementedError(
          "Metal direct AIR while accumulator requires a positive step.");
    }

    const int32_t init_iter_value = init_iter->literal().Get<int32_t>({});
    const float init_acc_value = init_acc->literal().Get<float>({});
    const std::string header = NewLabel("while_header");
    const std::string loop_body = NewLabel("while_body");
    const std::string done = NewLabel("while_done");
    const std::string iter = NewName("while_iter");
    const std::string acc = NewName("while_acc");
    const std::string in_range = NewName("while_in_range");
    const std::string iter64 = NewName("while_iter64");
    const std::string next_acc = NewName("while_next_acc");
    const std::string next_iter = NewName("while_next_iter");

    body->push_back(absl::StrFormat("  br label %%%s", header));
    body->push_back(absl::StrFormat("%s:", header));
    body->push_back(absl::StrFormat(
        "  %s = phi i32 [ %d, %%body ], [ %s, %%%s ]", iter,
        init_iter_value, next_iter, loop_body));
    body->push_back(absl::StrFormat(
        "  %s = phi float [ %s, %%body ], [ %s, %%%s ]", acc,
        FloatLiteral(init_acc_value), next_acc, loop_body));
    body->push_back(absl::StrFormat("  %s = icmp slt i32 %s, %d", in_range,
                                    iter, limit));
    body->push_back(absl::StrFormat("  br i1 %s, label %%%s, label %%%s",
                                    in_range, loop_body, done));

    body->push_back(absl::StrFormat("%s:", loop_body));
    body->push_back(absl::StrFormat("  %s = sext i32 %s to i64", iter64,
                                    iter));
    TF_ASSIGN_OR_RETURN(std::string value,
                        EmitLoadFromLinearIndex(input, iter64, body));
    body->push_back(absl::StrFormat("  %s = fadd fast float %s, %s",
                                    next_acc, acc, value));
    body->push_back(absl::StrFormat("  %s = add i32 %s, %d", next_iter, iter,
                                    step));
    body->push_back(absl::StrFormat("  br label %%%s", header));

    body->push_back(absl::StrFormat("%s:", done));
    return acc;
  }

  absl::StatusOr<std::string> EmitSimpleWhileVectorAccumulator(
      const HloInstruction* instr, std::vector<std::string>* body) {
    const HloInstruction* while_instr = instr->operand(0);
    if (while_instr->operand_count() != 1 ||
        while_instr->operand(0)->opcode() != HloOpcode::kTuple ||
        while_instr->operand(0)->operand_count() != 3) {
      return absl::UnimplementedError(
          "Metal direct AIR vector while requires a narrow loop tuple.");
    }

    const HloInstruction* init_tuple = while_instr->operand(0);
    const HloInstruction* init_iter = init_tuple->operand(0);
    const HloInstruction* init_acc = init_tuple->operand(1);
    const HloInstruction* input = init_tuple->operand(2);
    if (!init_iter->IsConstant() || !IsScalarLikeS32(init_iter->shape()) ||
        !IsF32Array(input->shape()) ||
        !ShapeUtil::Equal(input->shape(), instr->shape())) {
      return absl::UnimplementedError(
          "Metal direct AIR vector while requires a constant scalar iteration "
          "start and a carried input matching the result shape.");
    }
    TF_ASSIGN_OR_RETURN(float init_acc_value,
                        ExtractF32ScalarConstant(init_acc));

    const HloInstruction* cond_root =
        while_instr->while_condition()->root_instruction();
    if (cond_root->opcode() != HloOpcode::kCompare ||
        cond_root->comparison_direction() != ComparisonDirection::kLt ||
        cond_root->operand(1)->opcode() != HloOpcode::kConstant ||
        !IsScalarLikeS32(cond_root->operand(1)->shape())) {
      return absl::UnimplementedError(
          "Metal direct AIR vector while accumulator requires an i < constant "
          "condition.");
    }
    const int32_t limit = cond_root->operand(1)->literal().Get<int32_t>({});

    const HloInstruction* body_root =
        while_instr->while_body()->root_instruction();
    if (body_root->opcode() != HloOpcode::kTuple ||
        body_root->operand_count() != 3 ||
        body_root->operand(0)->opcode() != HloOpcode::kAdd ||
        body_root->operand(1)->opcode() != HloOpcode::kAdd) {
      return absl::UnimplementedError(
          "Metal direct AIR vector while accumulator requires tuple(i + c, "
          "acc + input, input) body.");
    }
    TF_ASSIGN_OR_RETURN(int32_t step,
                        ExtractS32ConstantAddend(body_root->operand(0)));
    if (step <= 0) {
      return absl::UnimplementedError(
          "Metal direct AIR vector while accumulator requires a positive "
          "step.");
    }

    const int32_t init_iter_value = init_iter->literal().Get<int32_t>({});
    const std::string header = NewLabel("while_vector_header");
    const std::string loop_body = NewLabel("while_vector_body");
    const std::string done = NewLabel("while_vector_done");
    const std::string iter = NewName("while_vector_iter");
    const std::string acc = NewName("while_vector_acc");
    const std::string in_range = NewName("while_vector_in_range");
    const std::string next_acc = NewName("while_vector_next_acc");
    const std::string next_iter = NewName("while_vector_next_iter");

    body->push_back(absl::StrFormat("  br label %%%s", header));
    body->push_back(absl::StrFormat("%s:", header));
    body->push_back(absl::StrFormat(
        "  %s = phi i32 [ %d, %%body ], [ %s, %%%s ]", iter,
        init_iter_value, next_iter, loop_body));
    body->push_back(absl::StrFormat(
        "  %s = phi float [ %s, %%body ], [ %s, %%%s ]", acc,
        FloatLiteral(init_acc_value), next_acc, loop_body));
    body->push_back(absl::StrFormat("  %s = icmp slt i32 %s, %d", in_range,
                                    iter, limit));
    body->push_back(absl::StrFormat("  br i1 %s, label %%%s, label %%%s",
                                    in_range, loop_body, done));

    body->push_back(absl::StrFormat("%s:", loop_body));
    TF_ASSIGN_OR_RETURN(std::string value,
                        EmitLoadFromLinearIndex(input, "%idx", body));
    body->push_back(absl::StrFormat("  %s = fadd fast float %s, %s",
                                    next_acc, acc, value));
    body->push_back(absl::StrFormat("  %s = add i32 %s, %d", next_iter, iter,
                                    step));
    body->push_back(absl::StrFormat("  br label %%%s", header));

    body->push_back(absl::StrFormat("%s:", done));
    return acc;
  }

  absl::StatusOr<int32_t> ExtractS32ConstantAddend(
      const HloInstruction* instr) {
    if (instr->opcode() != HloOpcode::kAdd) {
      return absl::UnimplementedError(
          "Metal direct AIR expected an s32 add instruction.");
    }
    for (const HloInstruction* operand : instr->operands()) {
      if (operand->IsConstant() && IsScalarLikeS32(operand->shape())) {
        return operand->literal().Get<int32_t>({});
      }
    }
    return absl::UnimplementedError(
        "Metal direct AIR expected an s32 constant addend.");
  }

  absl::StatusOr<float> ExtractF32ScalarConstant(const HloInstruction* instr) {
    if (instr->IsConstant() && IsScalarLikeF32(instr->shape())) {
      return instr->literal().Get<float>({});
    }
    if (instr->opcode() == HloOpcode::kBroadcast) {
      return ExtractF32ScalarConstant(instr->operand(0));
    }
    return absl::UnimplementedError(
        "Metal direct AIR expected an f32 scalar constant.");
  }

  absl::StatusOr<std::string> EmitRank1OrderStatistic(
      const HloInstruction* input, int64_t input_elements, bool descending,
      std::vector<std::string>* body, bool return_index) {
    const PrimitiveType value_type = input->shape().element_type();
    if (value_type != F32 && value_type != S32 && value_type != U16) {
      return absl::UnimplementedError(
          "Metal direct AIR rank-1 ordering currently supports only f32, s32, "
          "and u16 inputs.");
    }
    if (input_elements <= 0) {
      return absl::UnimplementedError(
          "Metal direct AIR rank-1 ordering does not support empty inputs.");
    }

    const std::string outer = NewLabel("order_outer");
    const std::string candidate = NewLabel("order_candidate");
    const std::string inner = NewLabel("order_inner");
    const std::string inner_body = NewLabel("order_inner_body");
    const std::string after_inner = NewLabel("order_after_inner");
    const std::string done = NewLabel("order_done");
    const std::string j = NewName("order_j");
    const std::string selected = NewName("order_selected");
    const std::string j_in_range = NewName("order_j_in_range");
    const std::string j64 = NewName("order_j64");
    const std::string i = NewName("order_i");
    const std::string rank = NewName("order_rank");
    const std::string i_in_range = NewName("order_i_in_range");
    const std::string i64 = NewName("order_i64");
    const std::string ordered = NewName("order_ordered");
    const std::string equal = NewName("order_equal");
    const std::string tie_before = NewName("order_tie_before");
    const std::string stable_tie = NewName("order_stable_tie");
    const std::string before = NewName("order_before");
    const std::string rank_inc = NewName("order_rank_inc");
    const std::string rank_next = NewName("order_rank_next");
    const std::string i_next = NewName("order_i_next");
    const std::string has_output_rank = NewName("order_has_output_rank");
    const std::string selected_next = NewName("order_selected_next");
    const std::string j_next = NewName("order_j_next");
    const absl::string_view selected_type =
        return_index ? "i32" : ElementIrType(value_type);
    const std::string selected_init =
        (return_index || value_type == S32 || value_type == U16)
            ? "0"
            : "0x0000000000000000";

    body->push_back(absl::StrFormat("  br label %%%s", outer));
    body->push_back(absl::StrFormat("%s:", outer));
    body->push_back(absl::StrFormat(
        "  %s = phi i32 [ 0, %%body ], [ %s, %%%s ]", j, j_next,
        after_inner));
    body->push_back(absl::StrFormat(
        "  %s = phi %s [ %s, %%body ], [ %s, %%%s ]", selected,
        selected_type, selected_init, selected_next, after_inner));
    body->push_back(absl::StrFormat("  %s = icmp ult i32 %s, %d", j_in_range,
                                    j, input_elements));
    body->push_back(absl::StrFormat("  br i1 %s, label %%%s, label %%%s",
                                    j_in_range, candidate, done));

    body->push_back(absl::StrFormat("%s:", candidate));
    body->push_back(absl::StrFormat("  %s = zext i32 %s to i64", j64, j));
    TF_ASSIGN_OR_RETURN(std::string candidate_value,
                        EmitLoadFromLinearIndex(input, j64, body));
    body->push_back(absl::StrFormat("  br label %%%s", inner));

    body->push_back(absl::StrFormat("%s:", inner));
    body->push_back(absl::StrFormat(
        "  %s = phi i32 [ 0, %%%s ], [ %s, %%%s ]", i, candidate, i_next,
        inner_body));
    body->push_back(absl::StrFormat(
        "  %s = phi i32 [ 0, %%%s ], [ %s, %%%s ]", rank, candidate,
        rank_next, inner_body));
    body->push_back(absl::StrFormat("  %s = icmp ult i32 %s, %d", i_in_range,
                                    i, input_elements));
    body->push_back(absl::StrFormat("  br i1 %s, label %%%s, label %%%s",
                                    i_in_range, inner_body, after_inner));

    body->push_back(absl::StrFormat("%s:", inner_body));
    body->push_back(absl::StrFormat("  %s = zext i32 %s to i64", i64, i));
    TF_ASSIGN_OR_RETURN(std::string other_value,
                        EmitLoadFromLinearIndex(input, i64, body));
    if (value_type == S32 || value_type == U16) {
      const char* lt = value_type == S32 ? "slt" : "ult";
      const char* gt = value_type == S32 ? "sgt" : "ugt";
      body->push_back(absl::StrFormat(
          "  %s = icmp %s %s %s, %s", ordered, descending ? gt : lt,
          ElementIrType(value_type), other_value, candidate_value));
      body->push_back(absl::StrFormat("  %s = icmp eq %s %s, %s", equal,
                                      ElementIrType(value_type), other_value,
                                      candidate_value));
    } else {
      body->push_back(absl::StrFormat(
          "  %s = fcmp fast %s float %s, %s", ordered,
          descending ? "ogt" : "olt", other_value, candidate_value));
      body->push_back(absl::StrFormat(
          "  %s = fcmp fast oeq float %s, %s", equal, other_value,
          candidate_value));
    }
    body->push_back(
        absl::StrFormat("  %s = icmp ult i32 %s, %s", tie_before, i, j));
    body->push_back(absl::StrFormat("  %s = and i1 %s, %s", stable_tie,
                                    equal, tie_before));
    body->push_back(absl::StrFormat("  %s = or i1 %s, %s", before, ordered,
                                    stable_tie));
    body->push_back(
        absl::StrFormat("  %s = zext i1 %s to i32", rank_inc, before));
    body->push_back(absl::StrFormat("  %s = add i32 %s, %s", rank_next, rank,
                                    rank_inc));
    body->push_back(absl::StrFormat("  %s = add i32 %s, 1", i_next, i));
    body->push_back(absl::StrFormat("  br label %%%s", inner));

    body->push_back(absl::StrFormat("%s:", after_inner));
    body->push_back(absl::StrFormat("  %s = icmp eq i32 %s, %%idx32",
                                    has_output_rank, rank));
    if (return_index) {
      body->push_back(absl::StrFormat(
          "  %s = select i1 %s, i32 %s, i32 %s", selected_next,
          has_output_rank, j, selected));
    } else {
      body->push_back(absl::StrFormat(
          "  %s = select i1 %s, %s %s, %s %s", selected_next,
          has_output_rank, selected_type, candidate_value, selected_type,
          selected));
    }
    body->push_back(absl::StrFormat("  %s = add i32 %s, 1", j_next, j));
    body->push_back(absl::StrFormat("  br label %%%s", outer));

    body->push_back(absl::StrFormat("%s:", done));
    return selected;
  }

  absl::StatusOr<std::string> EmitRank2OrderStatistic(
      const HloInstruction* input, int64_t rows, int64_t input_cols,
      int64_t output_cols, bool descending, std::vector<std::string>* body,
      bool return_index) {
    const PrimitiveType value_type = input->shape().element_type();
    if (value_type != F32 && value_type != S32 && value_type != U16) {
      return absl::UnimplementedError(
          "Metal direct AIR rank-2 ordering currently supports only f32, s32, "
          "and u16 inputs.");
    }
    if (rows <= 0 || input_cols <= 0 || output_cols <= 0 ||
        output_cols > input_cols) {
      return absl::UnimplementedError(
          "Metal direct AIR rank-2 ordering requires non-empty row-wise "
          "inputs and k <= columns.");
    }

    const std::string output_row = NewName("order2_output_row");
    const std::string output_rank = NewName("order2_output_rank");
    const std::string row64 = NewName("order2_row64");
    const std::string row_offset = NewName("order2_row_offset");
    body->push_back(absl::StrFormat("  %s = udiv i32 %%idx32, %d", output_row,
                                    output_cols));
    body->push_back(absl::StrFormat("  %s = urem i32 %%idx32, %d",
                                    output_rank, output_cols));
    body->push_back(
        absl::StrFormat("  %s = zext i32 %s to i64", row64, output_row));
    body->push_back(absl::StrFormat("  %s = mul i64 %s, %d", row_offset,
                                    row64, input_cols));

    const std::string outer = NewLabel("order2_outer");
    const std::string candidate = NewLabel("order2_candidate");
    const std::string inner = NewLabel("order2_inner");
    const std::string inner_body = NewLabel("order2_inner_body");
    const std::string after_inner = NewLabel("order2_after_inner");
    const std::string done = NewLabel("order2_done");
    const std::string j = NewName("order2_j");
    const std::string selected = NewName("order2_selected");
    const std::string j_in_range = NewName("order2_j_in_range");
    const std::string j64 = NewName("order2_j64");
    const std::string candidate_index = NewName("order2_candidate_index");
    const std::string i = NewName("order2_i");
    const std::string rank = NewName("order2_rank");
    const std::string i_in_range = NewName("order2_i_in_range");
    const std::string i64 = NewName("order2_i64");
    const std::string other_index = NewName("order2_other_index");
    const std::string ordered = NewName("order2_ordered");
    const std::string equal = NewName("order2_equal");
    const std::string tie_before = NewName("order2_tie_before");
    const std::string stable_tie = NewName("order2_stable_tie");
    const std::string before = NewName("order2_before");
    const std::string rank_inc = NewName("order2_rank_inc");
    const std::string rank_next = NewName("order2_rank_next");
    const std::string i_next = NewName("order2_i_next");
    const std::string has_output_rank = NewName("order2_has_output_rank");
    const std::string selected_next = NewName("order2_selected_next");
    const std::string j_next = NewName("order2_j_next");
    const absl::string_view selected_type =
        return_index ? "i32" : ElementIrType(value_type);
    const std::string selected_init =
        (return_index || value_type == S32 || value_type == U16)
            ? "0"
            : "0x0000000000000000";

    body->push_back(absl::StrFormat("  br label %%%s", outer));
    body->push_back(absl::StrFormat("%s:", outer));
    body->push_back(absl::StrFormat(
        "  %s = phi i32 [ 0, %%body ], [ %s, %%%s ]", j, j_next,
        after_inner));
    body->push_back(absl::StrFormat(
        "  %s = phi %s [ %s, %%body ], [ %s, %%%s ]", selected,
        selected_type, selected_init, selected_next, after_inner));
    body->push_back(absl::StrFormat("  %s = icmp ult i32 %s, %d", j_in_range,
                                    j, input_cols));
    body->push_back(absl::StrFormat("  br i1 %s, label %%%s, label %%%s",
                                    j_in_range, candidate, done));

    body->push_back(absl::StrFormat("%s:", candidate));
    body->push_back(absl::StrFormat("  %s = zext i32 %s to i64", j64, j));
    body->push_back(absl::StrFormat("  %s = add i64 %s, %s", candidate_index,
                                    row_offset, j64));
    TF_ASSIGN_OR_RETURN(std::string candidate_value,
                        EmitLoadFromLinearIndex(input, candidate_index, body));
    body->push_back(absl::StrFormat("  br label %%%s", inner));

    body->push_back(absl::StrFormat("%s:", inner));
    body->push_back(absl::StrFormat(
        "  %s = phi i32 [ 0, %%%s ], [ %s, %%%s ]", i, candidate, i_next,
        inner_body));
    body->push_back(absl::StrFormat(
        "  %s = phi i32 [ 0, %%%s ], [ %s, %%%s ]", rank, candidate,
        rank_next, inner_body));
    body->push_back(absl::StrFormat("  %s = icmp ult i32 %s, %d", i_in_range,
                                    i, input_cols));
    body->push_back(absl::StrFormat("  br i1 %s, label %%%s, label %%%s",
                                    i_in_range, inner_body, after_inner));

    body->push_back(absl::StrFormat("%s:", inner_body));
    body->push_back(absl::StrFormat("  %s = zext i32 %s to i64", i64, i));
    body->push_back(absl::StrFormat("  %s = add i64 %s, %s", other_index,
                                    row_offset, i64));
    TF_ASSIGN_OR_RETURN(std::string other_value,
                        EmitLoadFromLinearIndex(input, other_index, body));
    if (value_type == S32 || value_type == U16) {
      const char* lt = value_type == S32 ? "slt" : "ult";
      const char* gt = value_type == S32 ? "sgt" : "ugt";
      body->push_back(absl::StrFormat(
          "  %s = icmp %s %s %s, %s", ordered, descending ? gt : lt,
          ElementIrType(value_type), other_value, candidate_value));
      body->push_back(absl::StrFormat("  %s = icmp eq %s %s, %s", equal,
                                      ElementIrType(value_type), other_value,
                                      candidate_value));
    } else {
      body->push_back(absl::StrFormat(
          "  %s = fcmp fast %s float %s, %s", ordered,
          descending ? "ogt" : "olt", other_value, candidate_value));
      body->push_back(absl::StrFormat(
          "  %s = fcmp fast oeq float %s, %s", equal, other_value,
          candidate_value));
    }
    body->push_back(
        absl::StrFormat("  %s = icmp ult i32 %s, %s", tie_before, i, j));
    body->push_back(absl::StrFormat("  %s = and i1 %s, %s", stable_tie,
                                    equal, tie_before));
    body->push_back(absl::StrFormat("  %s = or i1 %s, %s", before, ordered,
                                    stable_tie));
    body->push_back(
        absl::StrFormat("  %s = zext i1 %s to i32", rank_inc, before));
    body->push_back(absl::StrFormat("  %s = add i32 %s, %s", rank_next, rank,
                                    rank_inc));
    body->push_back(absl::StrFormat("  %s = add i32 %s, 1", i_next, i));
    body->push_back(absl::StrFormat("  br label %%%s", inner));

    body->push_back(absl::StrFormat("%s:", after_inner));
    body->push_back(absl::StrFormat("  %s = icmp eq i32 %s, %s",
                                    has_output_rank, rank, output_rank));
    if (return_index) {
      body->push_back(absl::StrFormat(
          "  %s = select i1 %s, i32 %s, i32 %s", selected_next,
          has_output_rank, j, selected));
    } else {
      body->push_back(absl::StrFormat(
          "  %s = select i1 %s, %s %s, %s %s", selected_next,
          has_output_rank, selected_type, candidate_value, selected_type,
          selected));
    }
    body->push_back(absl::StrFormat("  %s = add i32 %s, 1", j_next, j));
    body->push_back(absl::StrFormat("  br label %%%s", outer));

    body->push_back(absl::StrFormat("%s:", done));
    return selected;
  }

  absl::StatusOr<std::string> EmitSlice(const HloInstruction* instr,
                                        bool force_scalar,
                                        std::vector<std::string>* body) {
    const int64_t slice_elements = ShapeUtil::ElementsIn(instr->shape());
    if ((!force_scalar &&
         slice_elements != ShapeUtil::ElementsIn(result_shape_)) ||
        (force_scalar && slice_elements != 1)) {
      return absl::UnimplementedError(
          "Metal direct AIR elementwise slice must have the same element count "
          "as the final result, unless used as a scalar operand.");
    }

    TF_ASSIGN_OR_RETURN(
        std::string source_index,
        EmitSliceSourceIndex(instr, force_scalar ? "0" : "%idx", body));
    return EmitLoadFromLinearIndex(instr->operand(0), source_index, body);
  }

  absl::StatusOr<std::string> EmitDynamicSlice(
      const HloInstruction* instr, std::vector<std::string>* body) {
    if (ShapeUtil::ElementsIn(instr->shape()) !=
        ShapeUtil::ElementsIn(result_shape_)) {
      return absl::UnimplementedError(
          "Metal direct AIR dynamic-slice result must have the same element "
          "count as the final result.");
    }
    TF_ASSIGN_OR_RETURN(std::string source_index,
                        EmitDynamicSliceSourceIndex(instr, "%idx", body));
    return EmitLoadFromLinearIndex(instr->operand(0), source_index, body);
  }

  absl::StatusOr<std::string> EmitReverse(const HloInstruction* instr,
                                          std::vector<std::string>* body) {
    if (ShapeUtil::ElementsIn(instr->shape()) !=
        ShapeUtil::ElementsIn(result_shape_)) {
      return absl::UnimplementedError(
          "Metal direct AIR reverse result must have the same element count "
          "as the final result.");
    }
    TF_ASSIGN_OR_RETURN(std::string source_index,
                        EmitReverseSourceIndex(instr, "%idx", body));
    return EmitLoadFromLinearIndex(instr->operand(0), source_index, body);
  }

  absl::StatusOr<std::string> EmitIota(const HloInstruction* instr,
                                       std::vector<std::string>* body) {
    if ((!IsF32Array(instr->shape()) && !IsS32Array(instr->shape())) ||
        ShapeUtil::ElementsIn(instr->shape()) !=
            ShapeUtil::ElementsIn(result_shape_)) {
      return absl::UnimplementedError(
          "Metal direct AIR elementwise iota currently supports only f32/s32 "
          "iotas with the final result element count.");
    }
    const int64_t rank = instr->shape().dimensions().size();
    const auto* iota = Cast<HloIotaInstruction>(instr);
    if (rank != 1 && rank != 2) {
      return absl::UnimplementedError(
          "Metal direct AIR elementwise iota currently supports only rank-1 "
          "and rank-2 iota results.");
    }
    std::string source_index;
    if (rank == 1) {
      if (iota->iota_dimension() != 0) {
        return absl::UnimplementedError(
            "Metal direct AIR rank-1 iota dimension must be 0.");
      }
      source_index = "%idx";
    } else {
      if (iota->iota_dimension() < 0 || iota->iota_dimension() > 1) {
        return absl::UnimplementedError(
            "Metal direct AIR rank-2 iota dimension must be 0 or 1.");
      }
      const int64_t minor = instr->shape().dimensions(1);
      source_index =
          NewName(iota->iota_dimension() == 0 ? "iota_row" : "iota_col");
      if (iota->iota_dimension() == 0) {
        body->push_back(
            absl::StrFormat("  %s = udiv i64 %%idx, %d", source_index, minor));
      } else {
        body->push_back(
            absl::StrFormat("  %s = urem i64 %%idx, %d", source_index, minor));
      }
    }
    if (instr->shape().element_type() == S32) {
      if (rank == 1) {
        return "%idx32";
      }
      std::string value = NewName("iota_s32");
      body->push_back(
          absl::StrFormat("  %s = trunc i64 %s to i32", value, source_index));
      return value;
    }
    std::string value = NewName("iota");
    body->push_back(absl::StrFormat("  %s = uitofp i64 %s to float", value,
                                    source_index));
    return value;
  }

  absl::StatusOr<std::string> EmitBroadcastFromLinearIndex(
      const HloInstruction* instr, absl::string_view linear_index,
      std::vector<std::string>* body) {
    const HloInstruction* operand = instr->operand(0);
    if (!instr->shape().IsArray() || !operand->shape().IsArray() ||
        instr->dimensions().size() != operand->shape().dimensions().size() ||
        ShapeUtil::ElementsIn(instr->shape()) == 0) {
      return absl::UnimplementedError(
          "Metal direct AIR elementwise broadcast requires array operands, "
          "valid broadcast dimensions, and non-empty result shapes.");
    }

    const int64_t result_rank = instr->shape().dimensions().size();
    const int64_t operand_rank = operand->shape().dimensions().size();
    if (operand_rank == 0) {
      return EmitLoadFromLinearIndex(operand, "0", body);
    }

    std::vector<int64_t> result_strides(result_rank, 1);
    for (int64_t dim = result_rank - 2; dim >= 0; --dim) {
      result_strides[dim] =
          result_strides[dim + 1] * instr->shape().dimensions(dim + 1);
    }
    std::vector<int64_t> operand_strides(operand_rank, 1);
    for (int64_t dim = operand_rank - 2; dim >= 0; --dim) {
      operand_strides[dim] =
          operand_strides[dim + 1] * operand->shape().dimensions(dim + 1);
    }

    std::string source_index = "0";
    for (int64_t operand_dim = 0; operand_dim < operand_rank; ++operand_dim) {
      const int64_t result_dim = instr->dimensions()[operand_dim];
      if (result_dim < 0 || result_dim >= result_rank ||
          operand->shape().dimensions(operand_dim) !=
              instr->shape().dimensions(result_dim)) {
        return absl::UnimplementedError(
            "Metal direct AIR broadcast operand dimension mismatch.");
      }

      std::string coord = "0";
      const int64_t result_dim_size = instr->shape().dimensions(result_dim);
      if (result_dim_size != 1) {
        coord = std::string(linear_index);
        if (result_strides[result_dim] != 1) {
          coord = NewName("broadcast_div");
          body->push_back(absl::StrFormat("  %s = udiv i64 %s, %d", coord,
                                          linear_index,
                                          result_strides[result_dim]));
        }
        if (result_dim_size != 0) {
          std::string rem = NewName("broadcast_coord");
          body->push_back(absl::StrFormat("  %s = urem i64 %s, %d", rem,
                                          coord, result_dim_size));
          coord = rem;
        }
      }

      std::string term = coord;
      if (operand_strides[operand_dim] != 1 && coord != "0") {
        term = NewName("broadcast_term");
        body->push_back(absl::StrFormat("  %s = mul i64 %s, %d", term, coord,
                                        operand_strides[operand_dim]));
      }
      if (source_index == "0") {
        source_index = term;
      } else if (term != "0") {
        std::string sum = NewName("broadcast_index");
        body->push_back(absl::StrFormat("  %s = add i64 %s, %s", sum,
                                        source_index, term));
        source_index = sum;
      }
    }
    return EmitLoadFromLinearIndex(operand, source_index, body);
  }

  absl::StatusOr<std::string> EmitBroadcast(
      const HloInstruction* instr, std::vector<std::string>* body) {
    if (ShapeUtil::ElementsIn(instr->shape()) !=
        ShapeUtil::ElementsIn(result_shape_)) {
      return absl::UnimplementedError(
          "Metal direct AIR root broadcast must match the result element "
          "count.");
    }
    return EmitBroadcastFromLinearIndex(instr, "%idx", body);
  }

  absl::StatusOr<std::string> EmitDynamicUpdateSlice(
      const HloInstruction* instr, std::vector<std::string>* body) {
    const HloInstruction* input = instr->operand(0);
    const HloInstruction* update = instr->operand(1);
    const PrimitiveType element_type = instr->shape().element_type();
    if (!IsSupportedElementwiseArray(instr->shape()) ||
        !IsSupportedElementwiseArray(input->shape()) ||
        !IsSupportedElementwiseArray(update->shape()) ||
        input->shape().element_type() != element_type ||
        update->shape().element_type() != element_type ||
        !ShapeUtil::Equal(instr->shape(), input->shape()) ||
        !ShapeUtil::Equal(instr->shape(), result_shape_)) {
      return absl::UnimplementedError(
          "Metal direct AIR dynamic-update-slice currently supports only "
          "supported array updates with result shape matching the input "
          "shape.");
    }
    const int64_t rank = instr->shape().dimensions().size();
    if (rank != update->shape().dimensions().size() ||
        instr->operand_count() != rank + 2 ||
        (rank != 1 && rank != 2 && rank != 3)) {
      return absl::UnimplementedError(
          "Metal direct AIR dynamic-update-slice currently supports only "
          "rank-1, rank-2, and rank-3 updates.");
    }

    std::vector<std::string> starts;
    starts.reserve(rank);
    for (int64_t dim = 0; dim < rank; ++dim) {
      const HloInstruction* start = instr->operand(dim + 2);
      if (!IsScalarLikeIndex(start->shape())) {
        return absl::UnimplementedError(
            "Metal direct AIR dynamic-update-slice requires scalar integer "
            "start indices.");
      }
      const int64_t max_start =
          instr->shape().dimensions(dim) - update->shape().dimensions(dim);
      TF_ASSIGN_OR_RETURN(std::string raw, EmitStartIndexAsS32(start, body));
      starts.push_back(EmitClampedStartIndex(raw, max_start, body));
    }

    TF_ASSIGN_OR_RETURN(std::string input_value,
                        EmitLoadFromLinearIndex(input, "%idx", body));
    std::string in_update;
    std::string update_index;
    if (rank == 1) {
      std::string ge_start = NewName("dus_ge_start");
      std::string end = NewName("dus_end");
      std::string lt_end = NewName("dus_lt_end");
      in_update = NewName("dus_in_update");
      body->push_back(absl::StrFormat("  %s = icmp uge i64 %%idx, %s",
                                      ge_start, starts[0]));
      body->push_back(absl::StrFormat("  %s = add i64 %s, %d", end,
                                      starts[0],
                                      update->shape().dimensions(0)));
      body->push_back(absl::StrFormat("  %s = icmp ult i64 %%idx, %s",
                                      lt_end, end));
      body->push_back(absl::StrFormat("  %s = and i1 %s, %s", in_update,
                                      ge_start, lt_end));
      update_index = NewName("dus_update_index");
      body->push_back(absl::StrFormat("  %s = sub i64 %%idx, %s",
                                      update_index, starts[0]));
    } else if (rank == 2) {
      const int64_t result_minor = instr->shape().dimensions(1);
      const int64_t update_minor = update->shape().dimensions(1);
      std::string row = NewName("dus_row");
      std::string col = NewName("dus_col");
      body->push_back(
          absl::StrFormat("  %s = udiv i64 %%idx, %d", row, result_minor));
      body->push_back(
          absl::StrFormat("  %s = urem i64 %%idx, %d", col, result_minor));
      std::string row_ge_start = NewName("dus_row_ge_start");
      std::string row_end = NewName("dus_row_end");
      std::string row_lt_end = NewName("dus_row_lt_end");
      std::string col_ge_start = NewName("dus_col_ge_start");
      std::string col_end = NewName("dus_col_end");
      std::string col_lt_end = NewName("dus_col_lt_end");
      std::string row_in = NewName("dus_row_in");
      std::string col_in = NewName("dus_col_in");
      in_update = NewName("dus_in_update");
      body->push_back(absl::StrFormat("  %s = icmp uge i64 %s, %s",
                                      row_ge_start, row, starts[0]));
      body->push_back(absl::StrFormat("  %s = add i64 %s, %d", row_end,
                                      starts[0],
                                      update->shape().dimensions(0)));
      body->push_back(absl::StrFormat("  %s = icmp ult i64 %s, %s",
                                      row_lt_end, row, row_end));
      body->push_back(absl::StrFormat("  %s = icmp uge i64 %s, %s",
                                      col_ge_start, col, starts[1]));
      body->push_back(absl::StrFormat("  %s = add i64 %s, %d", col_end,
                                      starts[1],
                                      update->shape().dimensions(1)));
      body->push_back(absl::StrFormat("  %s = icmp ult i64 %s, %s",
                                      col_lt_end, col, col_end));
      body->push_back(absl::StrFormat("  %s = and i1 %s, %s", row_in,
                                      row_ge_start, row_lt_end));
      body->push_back(absl::StrFormat("  %s = and i1 %s, %s", col_in,
                                      col_ge_start, col_lt_end));
      body->push_back(absl::StrFormat("  %s = and i1 %s, %s", in_update,
                                      row_in, col_in));
      std::string local_row = NewName("dus_local_row");
      std::string local_col = NewName("dus_local_col");
      std::string row_offset = NewName("dus_row_offset");
      update_index = NewName("dus_update_index");
      body->push_back(absl::StrFormat("  %s = sub i64 %s, %s", local_row, row,
                                      starts[0]));
      body->push_back(absl::StrFormat("  %s = sub i64 %s, %s", local_col, col,
                                      starts[1]));
      body->push_back(absl::StrFormat("  %s = mul i64 %s, %d", row_offset,
                                      local_row, update_minor));
      body->push_back(absl::StrFormat("  %s = add i64 %s, %s", update_index,
                                      row_offset, local_col));
    } else {
      const int64_t result_dim1 = instr->shape().dimensions(1);
      const int64_t result_dim2 = instr->shape().dimensions(2);
      const int64_t update_dim1 = update->shape().dimensions(1);
      const int64_t update_dim2 = update->shape().dimensions(2);
      std::string plane = NewName("dus_plane");
      std::string dim0 = NewName("dus_dim0");
      std::string dim1 = NewName("dus_dim1");
      std::string dim2 = NewName("dus_dim2");
      body->push_back(absl::StrFormat("  %s = udiv i64 %%idx, %d", dim0,
                                      result_dim1 * result_dim2));
      body->push_back(absl::StrFormat("  %s = urem i64 %%idx, %d", plane,
                                      result_dim1 * result_dim2));
      body->push_back(
          absl::StrFormat("  %s = udiv i64 %s, %d", dim1, plane, result_dim2));
      body->push_back(
          absl::StrFormat("  %s = urem i64 %s, %d", dim2, plane, result_dim2));

      std::vector<std::string> dim_values = {dim0, dim1, dim2};
      std::vector<std::string> in_dims;
      in_dims.reserve(3);
      for (int64_t dim = 0; dim < 3; ++dim) {
        std::string ge_start = NewName("dus_ge_start");
        std::string end = NewName("dus_end");
        std::string lt_end = NewName("dus_lt_end");
        std::string dim_in = NewName("dus_dim_in");
        body->push_back(absl::StrFormat("  %s = icmp uge i64 %s, %s",
                                        ge_start, dim_values[dim],
                                        starts[dim]));
        body->push_back(absl::StrFormat("  %s = add i64 %s, %d", end,
                                        starts[dim],
                                        update->shape().dimensions(dim)));
        body->push_back(absl::StrFormat("  %s = icmp ult i64 %s, %s",
                                        lt_end, dim_values[dim], end));
        body->push_back(absl::StrFormat("  %s = and i1 %s, %s", dim_in,
                                        ge_start, lt_end));
        in_dims.push_back(dim_in);
      }
      std::string dim01_in = NewName("dus_dim01_in");
      in_update = NewName("dus_in_update");
      body->push_back(absl::StrFormat("  %s = and i1 %s, %s", dim01_in,
                                      in_dims[0], in_dims[1]));
      body->push_back(absl::StrFormat("  %s = and i1 %s, %s", in_update,
                                      dim01_in, in_dims[2]));

      std::string local_dim0 = NewName("dus_local_dim0");
      std::string local_dim1 = NewName("dus_local_dim1");
      std::string local_dim2 = NewName("dus_local_dim2");
      body->push_back(absl::StrFormat("  %s = sub i64 %s, %s", local_dim0,
                                      dim0, starts[0]));
      body->push_back(absl::StrFormat("  %s = sub i64 %s, %s", local_dim1,
                                      dim1, starts[1]));
      body->push_back(absl::StrFormat("  %s = sub i64 %s, %s", local_dim2,
                                      dim2, starts[2]));
      std::string dim0_offset = NewName("dus_dim0_offset");
      std::string dim1_offset = NewName("dus_dim1_offset");
      std::string base_index = NewName("dus_base_index");
      update_index = NewName("dus_update_index");
      body->push_back(absl::StrFormat("  %s = mul i64 %s, %d", dim0_offset,
                                      local_dim0, update_dim1 * update_dim2));
      body->push_back(absl::StrFormat("  %s = mul i64 %s, %d", dim1_offset,
                                      local_dim1, update_dim2));
      body->push_back(absl::StrFormat("  %s = add i64 %s, %s", base_index,
                                      dim0_offset, dim1_offset));
      body->push_back(absl::StrFormat("  %s = add i64 %s, %s", update_index,
                                      base_index, local_dim2));
    }

    std::string safe_update_index = NewName("dus_safe_update_index");
    body->push_back(absl::StrFormat(
        "  %s = select i1 %s, i64 %s, i64 0", safe_update_index, in_update,
        update_index));
    TF_ASSIGN_OR_RETURN(
        std::string update_value,
        EmitLoadFromLinearIndex(update, safe_update_index, body));
    std::string result = NewName("dus_select");
    const char* ir_type = ValueIrType(element_type);
    body->push_back(absl::StrFormat(
        "  %s = select i1 %s, %s %s, %s %s", result, in_update, ir_type,
        update_value, ir_type, input_value));
    return result;
  }

  absl::StatusOr<std::string> EmitScatter(const HloInstruction* instr,
                                          std::vector<std::string>* body) {
    const HloInstruction* input = instr->operand(0);
    const HloInstruction* indices = instr->operand(1);
    const HloInstruction* updates = instr->operand(2);
    const ScatterDimensionNumbers& dnums = instr->scatter_dimension_numbers();
    const int64_t rank = input->shape().dimensions().size();
    const int64_t update_count = ShapeUtil::ElementsIn(updates->shape());
    if (!IsF32Array(instr->shape()) || !IsF32Array(input->shape()) ||
        !IsF32Array(updates->shape()) ||
        instr->shape().dimensions().size() != rank ||
        (rank != 1 && rank != 2) ||
        updates->shape().dimensions().size() != 1 ||
        indices->shape().element_type() != S32 ||
        indices->shape().dimensions().size() != 2 ||
        indices->shape().dimensions(1) != rank ||
        !ShapeUtil::Equal(instr->shape(), input->shape()) ||
        !ShapeUtil::Equal(instr->shape(), result_shape_) ||
        dnums.update_window_dims_size() != 0 ||
        dnums.inserted_window_dims_size() != rank ||
        dnums.scatter_dims_to_operand_dims_size() != rank ||
        dnums.index_vector_dim() != 1 ||
        update_count != indices->shape().dimensions(0)) {
      return absl::UnimplementedError(
          "Metal direct AIR scatter currently supports only rank-1/rank-2 f32 "
          "scatter set/add with scalar updates.");
    }
    for (int64_t dim = 0; dim < rank; ++dim) {
      if (dnums.inserted_window_dims(dim) != dim ||
          dnums.scatter_dims_to_operand_dims(dim) != dim) {
        return absl::UnimplementedError(
            "Metal direct AIR scatter currently requires identity scatter "
            "dimension mappings.");
      }
    }

    const HloInstruction* reducer = instr->to_apply()->root_instruction();
    const bool scatter_add = reducer->opcode() == HloOpcode::kAdd;
    const bool scatter_set =
        reducer->opcode() == HloOpcode::kParameter &&
        reducer->parameter_number() == 1;
    if (!scatter_add && !scatter_set) {
      return absl::UnimplementedError(
          "Metal direct AIR scatter currently supports only set and add "
          "update computations.");
    }

    TF_ASSIGN_OR_RETURN(std::string selected,
                        EmitLoadFromLinearIndex(input, "%idx", body));

    auto apply_update = [&](int64_t update_index,
                            absl::string_view target_index,
                            absl::string_view in_bounds) -> absl::Status {
      TF_ASSIGN_OR_RETURN(
          std::string update_value,
          EmitLoadFromLinearIndex(updates, absl::StrCat(update_index), body));
      std::string replacement = update_value;
      if (scatter_add) {
        replacement = EmitOp("fadd fast float", selected, update_value, body);
      }
      std::string is_index = NewName("scatter_is_index");
      std::string applies = is_index;
      std::string next = NewName("scatter_select");
      body->push_back(absl::StrFormat("  %s = icmp eq i64 %%idx, %s",
                                      is_index, target_index));
      if (in_bounds != "true") {
        applies = NewName("scatter_applies");
        body->push_back(absl::StrFormat("  %s = and i1 %s, %s", applies,
                                        is_index, in_bounds));
      }
      body->push_back(absl::StrFormat(
          "  %s = select i1 %s, float %s, float %s", next, applies,
          replacement, selected));
      selected = next;
      return absl::OkStatus();
    };

    auto static_indices_or = EvaluateS32Array(indices);
    if (static_indices_or.ok()) {
      const std::vector<int64_t>& static_indices = *static_indices_or;
      if (static_indices.size() != update_count * rank) {
        return absl::InvalidArgumentError(
            "Metal direct AIR scatter index count does not match updates.");
      }
      for (int64_t update_index = 0; update_index < update_count;
           ++update_index) {
        bool in_bounds = true;
        int64_t target_index = 0;
        for (int64_t dim = 0; dim < rank; ++dim) {
          const int64_t index = static_indices[update_index * rank + dim];
          in_bounds &= index >= 0 && index < input->shape().dimensions(dim);
          if (dim != 0) {
            target_index *= input->shape().dimensions(dim);
          }
          target_index += index;
        }
        if (!in_bounds) {
          continue;
        }
        TF_RETURN_IF_ERROR(
            apply_update(update_index, absl::StrCat(target_index), "true"));
      }
      return selected;
    }

    for (int64_t update_index = 0; update_index < update_count;
         ++update_index) {
      std::string target_index;
      std::string in_bounds = "true";
      for (int64_t dim = 0; dim < rank; ++dim) {
        TF_ASSIGN_OR_RETURN(
            std::string raw_index,
            EmitLoadFromLinearIndex(
                indices, absl::StrCat(update_index * rank + dim), body));
        std::string ge_zero = NewName("scatter_ge_zero");
        std::string lt_dim = NewName("scatter_lt_dim");
        std::string dim_in_bounds = NewName("scatter_dim_in_bounds");
        body->push_back(absl::StrFormat("  %s = icmp sge i32 %s, 0", ge_zero,
                                        raw_index));
        body->push_back(absl::StrFormat("  %s = icmp slt i32 %s, %d", lt_dim,
                                        raw_index,
                                        input->shape().dimensions(dim)));
        body->push_back(absl::StrFormat("  %s = and i1 %s, %s",
                                        dim_in_bounds, ge_zero, lt_dim));
        if (in_bounds == "true") {
          in_bounds = dim_in_bounds;
        } else {
          std::string combined = NewName("scatter_in_bounds");
          body->push_back(absl::StrFormat("  %s = and i1 %s, %s", combined,
                                          in_bounds, dim_in_bounds));
          in_bounds = combined;
        }

        std::string index_i64 = NewName("scatter_index_i64");
        body->push_back(absl::StrFormat("  %s = sext i32 %s to i64",
                                        index_i64, raw_index));
        if (dim == 0) {
          target_index = index_i64;
        } else {
          std::string scaled = NewName("scatter_scaled_index");
          std::string next_index = NewName("scatter_target_index");
          body->push_back(absl::StrFormat("  %s = mul i64 %s, %d", scaled,
                                          target_index,
                                          input->shape().dimensions(dim)));
          body->push_back(absl::StrFormat("  %s = add i64 %s, %s",
                                          next_index, scaled, index_i64));
          target_index = next_index;
        }
      }
      TF_RETURN_IF_ERROR(apply_update(update_index, target_index, in_bounds));
    }
    return selected;
  }

  absl::StatusOr<std::string> EmitReduceWindow(
      const HloInstruction* instr, std::vector<std::string>* body) {
    const HloInstruction* input = instr->operand(0);
    const int64_t rank = instr->shape().dimensions().size();
    const PrimitiveType reduce_type = instr->shape().element_type();
    const bool supported_reduce_type =
        IsFloatAccumulatorElementType(reduce_type) ||
        IsIntegerElementType(reduce_type);
    if (!supported_reduce_type ||
        input->shape().element_type() != reduce_type ||
        instr->operand(1)->shape().element_type() != reduce_type ||
        instr->operand_count() != 2 ||
        (rank != 1 && rank != 2 && rank != 3) ||
        input->shape().dimensions().size() != rank ||
        ShapeUtil::ElementsIn(instr->shape()) !=
            ShapeUtil::ElementsIn(result_shape_) ||
        instr->window().dimensions().size() != rank) {
      return absl::UnimplementedError(
          "Metal direct AIR reduce-window currently supports only "
          "rank-1/rank-2/rank-3 floating-point and integer results.");
    }
    for (int64_t i = 0; i < rank; ++i) {
      const WindowDimension& dim = instr->window().dimensions(i);
      if (dim.stride() != 1 || dim.padding_low() < 0 ||
          dim.padding_high() < 0 || dim.base_dilation() != 1 ||
          dim.window_dilation() != 1 || dim.window_reversal() ||
          instr->shape().dimensions(i) != input->shape().dimensions(i) +
                                              dim.padding_low() +
                                              dim.padding_high() - dim.size() +
                                              1) {
        return absl::UnimplementedError(
            "Metal direct AIR reduce-window currently supports only "
            "non-negative padded stride-1 non-dilated windows.");
      }
    }

    const HloComputation* reducer = instr->to_apply();
    if (reducer->num_parameters() != 2 ||
        reducer->parameter_instruction(0)->shape().element_type() !=
            reduce_type ||
        reducer->parameter_instruction(1)->shape().element_type() !=
            reduce_type ||
        !ShapeUtil::IsEffectiveScalar(
            reducer->parameter_instruction(0)->shape()) ||
        !ShapeUtil::IsEffectiveScalar(
            reducer->parameter_instruction(1)->shape())) {
      return absl::UnimplementedError(
          "Metal direct AIR reduce-window currently supports only binary "
          "scalar reducers with matching supported element types.");
    }

    const char* accumulator_ir_type = ValueIrType(reduce_type);
    TF_ASSIGN_OR_RETURN(std::string init_value,
                        EmitValue(instr->operand(1), /*force_scalar=*/true,
                                  body));
    auto apply_reducer = [&](absl::string_view accumulator,
                             absl::string_view value)
        -> absl::StatusOr<std::string> {
      ScalarParameterScope scope;
      scope.computation = reducer;
      scope.values[0] = std::string(accumulator);
      scope.values[1] = std::string(value);
      scalar_parameter_scopes_.push_back(std::move(scope));
      absl::Cleanup pop_scope = [this] {
        scalar_parameter_scopes_.pop_back();
      };
      return EmitValue(reducer->root_instruction(), /*force_scalar=*/true,
                       body);
    };
    auto offset_source = [&](absl::string_view out_coord, int64_t offset,
                             int64_t padding_low,
                             absl::string_view prefix) -> std::string {
      const int64_t delta = offset - padding_low;
      if (delta == 0) {
        return std::string(out_coord);
      }
      std::string source = NewName(prefix);
      if (delta > 0) {
        body->push_back(absl::StrFormat("  %s = add i64 %s, %d", source,
                                        out_coord, delta));
      } else {
        body->push_back(absl::StrFormat("  %s = sub i64 %s, %d", source,
                                        out_coord, -delta));
      }
      return source;
    };
    auto in_bounds = [&](absl::string_view source, int64_t bound,
                         absl::string_view prefix) -> std::string {
      std::string ge0 = NewName(absl::StrCat(prefix, "_ge0"));
      std::string lt = NewName(absl::StrCat(prefix, "_lt"));
      std::string ok = NewName(absl::StrCat(prefix, "_ok"));
      body->push_back(
          absl::StrFormat("  %s = icmp sge i64 %s, 0", ge0, source));
      body->push_back(
          absl::StrFormat("  %s = icmp slt i64 %s, %d", lt, source, bound));
      body->push_back(absl::StrFormat("  %s = and i1 %s, %s", ok, ge0, lt));
      return ok;
    };
    auto safe_index = [&](absl::string_view source,
                          absl::string_view predicate,
                          absl::string_view prefix) -> std::string {
      std::string safe = NewName(prefix);
      body->push_back(absl::StrFormat(
          "  %s = select i1 %s, i64 %s, i64 0", safe, predicate, source));
      return safe;
    };
    auto select_accumulator = [&](absl::string_view predicate,
                                  absl::string_view candidate,
                                  absl::string_view accumulator_value)
        -> std::string {
      std::string selected = NewName("reduce_window_accumulator");
      body->push_back(absl::StrFormat(
          "  %s = select i1 %s, %s %s, %s %s", selected, predicate,
          accumulator_ir_type, candidate, accumulator_ir_type,
          accumulator_value));
      return selected;
    };
    const bool init_is_negative_infinity =
        init_value == FloatLiteral(-std::numeric_limits<float>::infinity());
    auto select_identity_candidate = [&](absl::string_view accumulator_value,
                                         absl::string_view value,
                                         absl::string_view candidate)
        -> std::string {
      if (!init_is_negative_infinity) {
        return std::string(candidate);
      }
      std::string accumulator_bits = NewName("reduce_window_accumulator_bits");
      std::string is_init = NewName("reduce_window_accumulator_is_init");
      std::string selected = NewName("reduce_window_identity_candidate");
      body->push_back(absl::StrFormat("  %s = bitcast float %s to i32",
                                      accumulator_bits, accumulator_value));
      body->push_back(absl::StrFormat("  %s = icmp eq i32 %s, -8388608",
                                      is_init, accumulator_bits));
      body->push_back(absl::StrFormat(
          "  %s = select i1 %s, float %s, float %s", selected, is_init, value,
          candidate));
      return selected;
    };

    std::string accumulator = init_value;
    if (rank == 1) {
      const WindowDimension& dim = instr->window().dimensions(0);
      for (int64_t offset = 0; offset < dim.size(); ++offset) {
        std::string source = offset_source("%idx", offset, dim.padding_low(),
                                           "reduce_window_source");
        std::string ok =
            in_bounds(source, input->shape().dimensions(0), "reduce_window");
        std::string source_index =
            safe_index(source, ok, "reduce_window_safe_index");
        TF_ASSIGN_OR_RETURN(std::string value,
                            EmitLoadFromLinearIndex(input, source_index, body));
        TF_ASSIGN_OR_RETURN(std::string candidate,
                            apply_reducer(accumulator, value));
        candidate =
            select_identity_candidate(accumulator, value, candidate);
        accumulator = select_accumulator(ok, candidate, accumulator);
      }
      return accumulator;
    }

    if (rank == 3) {
      const int64_t result_dim1 = instr->shape().dimensions(1);
      const int64_t result_dim2 = instr->shape().dimensions(2);
      const int64_t input_dim1 = input->shape().dimensions(1);
      const int64_t input_dim2 = input->shape().dimensions(2);
      std::string out0 = NewName("reduce_window_dim0");
      std::string plane = NewName("reduce_window_plane");
      std::string out1 = NewName("reduce_window_dim1");
      std::string out2 = NewName("reduce_window_dim2");
      body->push_back(absl::StrFormat("  %s = udiv i64 %%idx, %d", out0,
                                      result_dim1 * result_dim2));
      body->push_back(absl::StrFormat("  %s = urem i64 %%idx, %d", plane,
                                      result_dim1 * result_dim2));
      body->push_back(
          absl::StrFormat("  %s = udiv i64 %s, %d", out1, plane, result_dim2));
      body->push_back(
          absl::StrFormat("  %s = urem i64 %s, %d", out2, plane, result_dim2));
      const WindowDimension& dim0 = instr->window().dimensions(0);
      const WindowDimension& dim1 = instr->window().dimensions(1);
      const WindowDimension& dim2 = instr->window().dimensions(2);
      for (int64_t offset0 = 0; offset0 < dim0.size(); ++offset0) {
        for (int64_t offset1 = 0; offset1 < dim1.size(); ++offset1) {
          for (int64_t offset2 = 0; offset2 < dim2.size(); ++offset2) {
            std::string source0 =
                offset_source(out0, offset0, dim0.padding_low(),
                              "reduce_window_source_dim0");
            std::string source1 =
                offset_source(out1, offset1, dim1.padding_low(),
                              "reduce_window_source_dim1");
            std::string source2 =
                offset_source(out2, offset2, dim2.padding_low(),
                              "reduce_window_source_dim2");
            std::string ok0 =
                in_bounds(source0, input->shape().dimensions(0),
                          "reduce_window_dim0");
            std::string ok1 =
                in_bounds(source1, input->shape().dimensions(1),
                          "reduce_window_dim1");
            std::string ok2 =
                in_bounds(source2, input->shape().dimensions(2),
                          "reduce_window_dim2");
            std::string ok01 = NewName("reduce_window_ok");
            std::string ok = NewName("reduce_window_ok");
            body->push_back(
                absl::StrFormat("  %s = and i1 %s, %s", ok01, ok0, ok1));
            body->push_back(
                absl::StrFormat("  %s = and i1 %s, %s", ok, ok01, ok2));
            source0 = safe_index(source0, ok, "reduce_window_safe_dim0");
            source1 = safe_index(source1, ok, "reduce_window_safe_dim1");
            source2 = safe_index(source2, ok, "reduce_window_safe_dim2");
            std::string term0 = NewName("reduce_window_term0");
            std::string term1 = NewName("reduce_window_term1");
            std::string base = NewName("reduce_window_base");
            std::string source_index = NewName("reduce_window_index");
            body->push_back(absl::StrFormat("  %s = mul i64 %s, %d", term0,
                                            source0, input_dim1 * input_dim2));
            body->push_back(absl::StrFormat("  %s = mul i64 %s, %d", term1,
                                            source1, input_dim2));
            body->push_back(absl::StrFormat("  %s = add i64 %s, %s", base,
                                            term0, term1));
            body->push_back(absl::StrFormat("  %s = add i64 %s, %s",
                                            source_index, base, source2));
            TF_ASSIGN_OR_RETURN(
                std::string value,
                EmitLoadFromLinearIndex(input, source_index, body));
            TF_ASSIGN_OR_RETURN(std::string candidate,
                                apply_reducer(accumulator, value));
            candidate =
                select_identity_candidate(accumulator, value, candidate);
            accumulator = select_accumulator(ok, candidate, accumulator);
          }
        }
      }
      return accumulator;
    }

    const int64_t result_cols = instr->shape().dimensions(1);
    const int64_t input_cols = input->shape().dimensions(1);
    std::string out_row = NewName("reduce_window_row");
    std::string out_col = NewName("reduce_window_col");
    body->push_back(
        absl::StrFormat("  %s = udiv i64 %%idx, %d", out_row, result_cols));
    body->push_back(
        absl::StrFormat("  %s = urem i64 %%idx, %d", out_col, result_cols));
    const WindowDimension& row_dim = instr->window().dimensions(0);
    const WindowDimension& col_dim = instr->window().dimensions(1);
    for (int64_t row_offset = 0; row_offset < row_dim.size(); ++row_offset) {
      for (int64_t col_offset = 0; col_offset < col_dim.size(); ++col_offset) {
        std::string source_row =
            offset_source(out_row, row_offset, row_dim.padding_low(),
                          "reduce_window_source_row");
        std::string source_col =
            offset_source(out_col, col_offset, col_dim.padding_low(),
                          "reduce_window_source_col");
        std::string row_ok =
            in_bounds(source_row, input->shape().dimensions(0),
                      "reduce_window_row");
        std::string col_ok =
            in_bounds(source_col, input->shape().dimensions(1),
                      "reduce_window_col");
        std::string ok = NewName("reduce_window_ok");
        body->push_back(
            absl::StrFormat("  %s = and i1 %s, %s", ok, row_ok, col_ok));
        source_row = safe_index(source_row, ok, "reduce_window_safe_row");
        source_col = safe_index(source_col, ok, "reduce_window_safe_col");
        std::string row_base = NewName("reduce_window_row_base");
        std::string source_index = NewName("reduce_window_index");
        body->push_back(absl::StrFormat("  %s = mul i64 %s, %d", row_base,
                                        source_row, input_cols));
        body->push_back(absl::StrFormat("  %s = add i64 %s, %s", source_index,
                                        row_base, source_col));
        TF_ASSIGN_OR_RETURN(std::string value,
                            EmitLoadFromLinearIndex(input, source_index, body));
        TF_ASSIGN_OR_RETURN(std::string candidate,
                            apply_reducer(accumulator, value));
        candidate =
            select_identity_candidate(accumulator, value, candidate);
        accumulator = select_accumulator(ok, candidate, accumulator);
      }
    }
    return accumulator;
  }

  int InputIndexForParameter(int64_t parameter_number, PrimitiveType type) {
    auto it = parameter_to_input_index_.find(parameter_number);
    if (it != parameter_to_input_index_.end()) {
      return it->second;
    }
    const int input_index = parameter_numbers_.size();
    parameter_to_input_index_[parameter_number] = input_index;
    parameter_numbers_.push_back(parameter_number);
    parameter_types_.push_back(type);
    return input_index;
  }

  absl::StatusOr<std::string> EmitCall(const HloInstruction* instr,
                                       bool force_scalar,
                                       std::vector<std::string>* body) {
    const HloComputation* callee = instr->to_apply();
    if (callee->num_parameters() != instr->operand_count()) {
      return absl::InvalidArgumentError(
          "Metal direct AIR elementwise call operand count does not match "
          "callee parameter count.");
    }
    CallParameterScope scope;
    scope.computation = callee;
    for (int64_t i = 0; i < instr->operand_count(); ++i) {
      scope.arguments[i] = instr->operand(i);
    }
    call_parameter_scopes_.push_back(std::move(scope));
    absl::Cleanup pop_scope = [this] { call_parameter_scopes_.pop_back(); };
    return EmitValue(callee->root_instruction(), force_scalar, body);
  }

  const HloInstruction* CallParameterOverride(
      const HloInstruction* instr) const {
    for (auto it = call_parameter_scopes_.rbegin();
         it != call_parameter_scopes_.rend(); ++it) {
      if (instr->parent() != it->computation) {
        continue;
      }
      auto argument = it->arguments.find(instr->parameter_number());
      if (argument != it->arguments.end()) {
        return argument->second;
      }
    }
    return nullptr;
  }

  std::optional<std::string> ScalarParameterOverride(
      const HloInstruction* instr) const {
    for (auto it = scalar_parameter_scopes_.rbegin();
         it != scalar_parameter_scopes_.rend(); ++it) {
      if (instr->parent() != it->computation) {
        continue;
      }
      auto value = it->values.find(instr->parameter_number());
      if (value != it->values.end()) {
        return value->second;
      }
    }
    return std::nullopt;
  }

  absl::StatusOr<std::string> EmitClamp(const HloInstruction* instr,
                                        std::vector<std::string>* body) {
    const PrimitiveType type = instr->shape().element_type();
    if (!IsSupportedElementwiseArray(instr->shape()) || type == PRED ||
        instr->operand_count() != 3) {
      return absl::UnimplementedError(
          "Metal direct AIR clamp currently supports only numeric elementwise "
          "arrays.");
    }
    for (const HloInstruction* operand : instr->operands()) {
      if (!operand->shape().IsArray() ||
          operand->shape().element_type() != type) {
        return absl::UnimplementedError(
            "Metal direct AIR clamp operands must have the result element "
            "type.");
      }
    }
    TF_ASSIGN_OR_RETURN(std::string min,
                        EmitValue(instr->operand(0), IsScalarOperand(instr, 0),
                                  body));
    TF_ASSIGN_OR_RETURN(std::string value,
                        EmitValue(instr->operand(1), IsScalarOperand(instr, 1),
                                  body));
    TF_ASSIGN_OR_RETURN(std::string max,
                        EmitValue(instr->operand(2), IsScalarOperand(instr, 2),
                                  body));
    switch (type) {
      case S8:
      case S16:
      case S32: {
        std::string lower_bounded =
            EmitIntCompareSelect("sgt", value, min, body, type);
        return EmitIntCompareSelect("slt", lower_bounded, max, body, type);
      }
      case U8:
      case U16:
      case U32: {
        std::string lower_bounded =
            EmitIntCompareSelect("ugt", value, min, body, type);
        return EmitIntCompareSelect("ult", lower_bounded, max, body, type);
      }
      case F16:
      case BF16:
      case F32: {
        std::string lower_bounded = EmitCompareSelect("ogt", value, min, body);
        return EmitCompareSelect("olt", lower_bounded, max, body);
      }
      default:
        return absl::UnimplementedError(
            "Metal direct AIR clamp does not support this element type.");
    }
  }

  absl::StatusOr<std::string> EmitCompare(const HloInstruction* instr,
                                          std::vector<std::string>* body) {
    Shape pred_shape = ShapeUtil::MakeShape(PRED, result_shape_.dimensions());
    if (!ShapeUtil::IsEffectiveScalar(instr->shape()) &&
        !ShapeUtil::Compatible(instr->shape(), pred_shape) &&
        (!instr->shape().IsArray() ||
         ShapeUtil::ElementsIn(instr->shape()) !=
             ShapeUtil::ElementsIn(result_shape_))) {
      return absl::UnimplementedError(
          "Metal direct AIR elementwise compare currently supports only "
          "predicates matching the result element count.");
    }
    TF_ASSIGN_OR_RETURN(std::string lhs,
                        EmitValue(instr->operand(0), IsScalarOperand(instr, 0),
                                  body));
    TF_ASSIGN_OR_RETURN(std::string rhs,
                        EmitValue(instr->operand(1), IsScalarOperand(instr, 1),
                                  body));
    std::string cmp = NewName("cmp");
    if (instr->operand(0)->shape().element_type() == PRED) {
      absl::string_view predicate;
      switch (instr->comparison_direction()) {
        case ComparisonDirection::kEq:
          predicate = "eq";
          break;
        case ComparisonDirection::kNe:
          predicate = "ne";
          break;
        default:
          return absl::UnimplementedError(
              "Metal direct AIR predicate compare supports only EQ and NE.");
      }
      body->push_back(absl::StrFormat("  %s = icmp %s i1 %s, %s", cmp,
                                      predicate, lhs, rhs));
    } else if (instr->operand(0)->shape().element_type() == S32) {
      absl::string_view predicate;
      switch (instr->comparison_direction()) {
        case ComparisonDirection::kEq:
          predicate = "eq";
          break;
        case ComparisonDirection::kNe:
          predicate = "ne";
          break;
        case ComparisonDirection::kGe:
          predicate = "sge";
          break;
        case ComparisonDirection::kGt:
          predicate = "sgt";
          break;
        case ComparisonDirection::kLe:
          predicate = "sle";
          break;
        case ComparisonDirection::kLt:
          predicate = "slt";
          break;
      }
      body->push_back(absl::StrFormat("  %s = icmp %s i32 %s, %s", cmp,
                                      predicate, lhs, rhs));
    } else {
      absl::string_view predicate;
      switch (instr->comparison_direction()) {
        case ComparisonDirection::kEq:
          predicate = "oeq";
          break;
        case ComparisonDirection::kNe:
          predicate = "one";
          break;
        case ComparisonDirection::kGe:
          predicate = "oge";
          break;
        case ComparisonDirection::kGt:
          predicate = "ogt";
          break;
        case ComparisonDirection::kLe:
          predicate = "ole";
          break;
        case ComparisonDirection::kLt:
          predicate = "olt";
          break;
      }
      body->push_back(absl::StrFormat("  %s = fcmp fast %s float %s, %s", cmp,
                                      predicate, lhs, rhs));
    }
    return cmp;
  }

  absl::StatusOr<std::string> EmitConvert(const HloInstruction* instr,
                                          std::vector<std::string>* body) {
    const PrimitiveType src_type = instr->operand(0)->shape().element_type();
    const PrimitiveType dst_type = instr->shape().element_type();
    TF_ASSIGN_OR_RETURN(
        std::string value,
        EmitValue(instr->operand(0), ShapeUtil::IsEffectiveScalar(instr->shape()),
                  body));
    if (src_type == dst_type) {
      return value;
    }
    if (IsFloatAccumulatorElementType(src_type) &&
        IsFloatAccumulatorElementType(dst_type)) {
      return value;
    }

    std::string converted = NewName("convert");
    if (src_type == PRED &&
        (dst_type == F32 || dst_type == F16 || dst_type == BF16)) {
      body->push_back(absl::StrFormat("  %s = uitofp i1 %s to float",
                                      converted, value));
      return converted;
    }
    if (src_type == PRED &&
        (dst_type == S8 || dst_type == U8 || dst_type == S16 ||
         dst_type == U16 || dst_type == S32 || dst_type == U32)) {
      const char* dst_ir_type = ElementIrType(dst_type);
      body->push_back(absl::StrFormat("  %s = zext i1 %s to %s", converted,
                                      value, dst_ir_type));
      return converted;
    }
    if (src_type == F32 && dst_type == S32) {
      body->push_back(absl::StrFormat("  %s = fptosi float %s to i32",
                                      converted, value));
      return converted;
    }
    if (src_type == S32 && dst_type == F32) {
      body->push_back(absl::StrFormat("  %s = sitofp i32 %s to float",
                                      converted, value));
      return converted;
    }
    if (src_type == S32 && dst_type == PRED) {
      body->push_back(
          absl::StrFormat("  %s = icmp ne i32 %s, 0", converted, value));
      return converted;
    }
    if (src_type == F32 && dst_type == PRED) {
      body->push_back(absl::StrFormat(
          "  %s = fcmp fast one float %s, 0x0000000000000000", converted,
          value));
      return converted;
    }
    return absl::UnimplementedError(absl::StrFormat(
        "Metal direct AIR elementwise convert does not support %s to %s.",
        primitive_util::LowercasePrimitiveTypeName(src_type),
        primitive_util::LowercasePrimitiveTypeName(dst_type)));
  }

  absl::StatusOr<std::string> EmitBitcastConvert(
      const HloInstruction* instr, std::vector<std::string>* body) {
    const Shape& src_shape = instr->operand(0)->shape();
    const PrimitiveType src_type = src_shape.element_type();
    const PrimitiveType dst_type = instr->shape().element_type();
    const int src_bit_width = ElementBitWidth(src_type);
    const int dst_bit_width = ElementBitWidth(dst_type);
    if (ShapeUtil::ElementsIn(src_shape) * src_bit_width !=
            ShapeUtil::ElementsIn(instr->shape()) * dst_bit_width ||
        ShapeUtil::ElementsIn(instr->shape()) !=
            ShapeUtil::ElementsIn(result_shape_)) {
      return absl::UnimplementedError(
          "Metal direct AIR bitcast-convert currently supports only "
          "shape-preserving bitcasts with matching total bit widths.");
    }

    if ((dst_type == S4 || dst_type == U4) &&
        ShapeUtil::Equal(instr->shape(), result_shape_)) {
      const HloInstruction* operand = instr->operand(0);
      if (operand->opcode() != HloOpcode::kParameter) {
        return absl::UnimplementedError(
            "Metal direct AIR bitcast-convert to s4/u4 currently supports "
            "only parameter operands.");
      }
      raw_byte_result_ = true;
      num_work_items_ = PackedByteSize(instr->shape());
      const int input_index =
          InputIndexForParameter(operand->parameter_number(), src_type);
      const char* src_ir_type = ElementIrType(src_type);
      std::string raw_arg = absl::StrFormat("%%arg%d", input_index);
      if (std::string(src_ir_type) != "i8") {
        raw_arg = NewName("bitcast_raw_arg");
        body->push_back(absl::StrFormat(
            "  %s = bitcast %s addrspace(1)* %%arg%d to i8 addrspace(1)*",
            raw_arg, src_ir_type, input_index));
      }
      std::string ptr = NewName("bitcast_ptr");
      std::string byte = NewName("bitcast_byte");
      body->push_back(absl::StrFormat(
          "  %s = getelementptr inbounds i8, i8 addrspace(1)* %s, i64 %%idx",
          ptr, raw_arg));
      body->push_back(absl::StrFormat(
          "  %s = load i8, i8 addrspace(1)* %s, align 1", byte, ptr));
      return byte;
    }

    if (dst_bit_width % 8 == 0) {
      const HloInstruction* operand = instr->operand(0);
      if (operand->opcode() == HloOpcode::kParameter) {
        const int input_index =
            InputIndexForParameter(operand->parameter_number(), src_type);
        const char* src_ir_type = ElementIrType(src_type);
        std::string raw_arg = absl::StrFormat("%%arg%d", input_index);
        if (std::string(src_ir_type) != "i8") {
          raw_arg = NewName("bitcast_raw_arg");
          body->push_back(absl::StrFormat(
              "  %s = bitcast %s addrspace(1)* %%arg%d to i8 addrspace(1)*",
              raw_arg, src_ir_type, input_index));
        }

        const int dst_bytes = dst_bit_width / 8;
        std::string byte_base = "%idx";
        if (dst_bytes != 1) {
          byte_base = NewName("bitcast_byte_base");
          body->push_back(absl::StrFormat("  %s = mul i64 %%idx, %d",
                                          byte_base, dst_bytes));
        }

        auto load_byte = [&](absl::string_view byte_index)
            -> absl::StatusOr<std::string> {
          std::string ptr = NewName("bitcast_ptr");
          std::string byte = NewName("bitcast_byte");
          body->push_back(absl::StrFormat(
              "  %s = getelementptr inbounds i8, i8 addrspace(1)* %s, i64 %s",
              ptr, raw_arg, byte_index));
          body->push_back(absl::StrFormat(
              "  %s = load i8, i8 addrspace(1)* %s, align 1", byte, ptr));
          return byte;
        };

        if (dst_bytes == 1) {
          return load_byte(byte_base);
        }

        const char* bits_type = dst_bytes == 2 ? "i16" : "i32";
        std::string bits;
        for (int byte_index = 0; byte_index < dst_bytes; ++byte_index) {
          std::string current_index = byte_base;
          if (byte_index != 0) {
            current_index = NewName("bitcast_byte_index");
            body->push_back(absl::StrFormat("  %s = add i64 %s, %d",
                                            current_index, byte_base,
                                            byte_index));
          }
          TF_ASSIGN_OR_RETURN(std::string byte, load_byte(current_index));
          std::string extended = NewName("bitcast_bits_part");
          body->push_back(absl::StrFormat("  %s = zext i8 %s to %s",
                                          extended, byte, bits_type));
          if (byte_index != 0) {
            std::string shifted = NewName("bitcast_bits_part");
            body->push_back(absl::StrFormat("  %s = shl %s %s, %d", shifted,
                                            bits_type, extended,
                                            byte_index * 8));
            extended = shifted;
          }
          if (bits.empty()) {
            bits = extended;
          } else {
            std::string combined = NewName("bitcast_bits");
            body->push_back(absl::StrFormat("  %s = or %s %s, %s", combined,
                                            bits_type, bits, extended));
            bits = combined;
          }
        }

        if (dst_type == F16) {
          std::string half_value = NewName("bitcast_half");
          std::string float_value = NewName("bitcast_float");
          body->push_back(absl::StrFormat("  %s = bitcast i16 %s to half",
                                          half_value, bits));
          body->push_back(absl::StrFormat("  %s = fpext half %s to float",
                                          float_value, half_value));
          return float_value;
        }
        if (dst_type == F32) {
          std::string float_value = NewName("bitcast_float");
          body->push_back(absl::StrFormat("  %s = bitcast i32 %s to float",
                                          float_value, bits));
          return float_value;
        }
        if (dst_type == BF16) {
          raw_bf16_result_ = true;
        }
        return bits;
      }
    }

    if ((src_type == S4 || src_type == U4) && dst_type == F16) {
      const HloInstruction* operand = instr->operand(0);
      if (operand->opcode() != HloOpcode::kParameter) {
        return absl::UnimplementedError(
            "Metal direct AIR s4/u4 to f16 bitcast-convert currently supports "
            "only parameter operands.");
      }
      const int input_index =
          InputIndexForParameter(operand->parameter_number(), src_type);
      std::string byte_base = NewName("bitcast_byte_base");
      std::string byte1_index = NewName("bitcast_byte1_index");
      std::string ptr0 = NewName("bitcast_ptr");
      std::string ptr1 = NewName("bitcast_ptr");
      std::string byte0 = NewName("bitcast_byte");
      std::string byte1 = NewName("bitcast_byte");
      std::string word0 = NewName("bitcast_word");
      std::string word1 = NewName("bitcast_word");
      std::string word1_shifted = NewName("bitcast_word");
      std::string bits = NewName("bitcast_bits");
      std::string half_value = NewName("bitcast_half");
      std::string float_value = NewName("bitcast_float");
      body->push_back(
          absl::StrFormat("  %s = mul i64 %%idx, 2", byte_base));
      body->push_back(
          absl::StrFormat("  %s = add i64 %s, 1", byte1_index, byte_base));
      body->push_back(absl::StrFormat(
          "  %s = getelementptr inbounds i8, i8 addrspace(1)* %%arg%d, i64 %s",
          ptr0, input_index, byte_base));
      body->push_back(absl::StrFormat(
          "  %s = getelementptr inbounds i8, i8 addrspace(1)* %%arg%d, i64 %s",
          ptr1, input_index, byte1_index));
      body->push_back(absl::StrFormat(
          "  %s = load i8, i8 addrspace(1)* %s, align 1", byte0, ptr0));
      body->push_back(absl::StrFormat(
          "  %s = load i8, i8 addrspace(1)* %s, align 1", byte1, ptr1));
      body->push_back(
          absl::StrFormat("  %s = zext i8 %s to i16", word0, byte0));
      body->push_back(
          absl::StrFormat("  %s = zext i8 %s to i16", word1, byte1));
      body->push_back(
          absl::StrFormat("  %s = shl i16 %s, 8", word1_shifted, word1));
      body->push_back(absl::StrFormat("  %s = or i16 %s, %s", bits, word0,
                                      word1_shifted));
      body->push_back(
          absl::StrFormat("  %s = bitcast i16 %s to half", half_value, bits));
      body->push_back(absl::StrFormat("  %s = fpext half %s to float",
                                      float_value, half_value));
      return float_value;
    }

    if (ElementTypeSize(src_type) != ElementTypeSize(dst_type) ||
        ShapeUtil::ElementsIn(src_shape) !=
            ShapeUtil::ElementsIn(instr->shape())) {
      return absl::UnimplementedError(
          "Metal direct AIR bitcast-convert currently supports only "
          "same-width elementwise bitcasts apart from packed s4/u4 to f16.");
    }

    TF_ASSIGN_OR_RETURN(
        std::string value,
        EmitValue(instr->operand(0), ShapeUtil::IsEffectiveScalar(instr->shape()),
                  body));
    if (src_type == U16 && dst_type == F16) {
      std::string half_value = NewName("bitcast_half");
      std::string float_value = NewName("bitcast_float");
      body->push_back(
          absl::StrFormat("  %s = bitcast i16 %s to half", half_value, value));
      body->push_back(absl::StrFormat("  %s = fpext half %s to float",
                                      float_value, half_value));
      return float_value;
    }
    if (src_type == F16 && dst_type == U16) {
      std::string half_value = NewName("bitcast_half");
      std::string bits = NewName("bitcast_bits");
      body->push_back(absl::StrFormat("  %s = fptrunc float %s to half",
                                      half_value, value));
      body->push_back(
          absl::StrFormat("  %s = bitcast half %s to i16", bits, half_value));
      return bits;
    }
    if (src_type == S32 && dst_type == F32) {
      std::string float_value = NewName("bitcast_float");
      body->push_back(
          absl::StrFormat("  %s = bitcast i32 %s to float", float_value,
                          value));
      return float_value;
    }
    if (src_type == F32 && dst_type == S32) {
      std::string bits = NewName("bitcast_bits");
      body->push_back(
          absl::StrFormat("  %s = bitcast float %s to i32", bits, value));
      return bits;
    }
    return absl::UnimplementedError(absl::StrFormat(
        "Metal direct AIR bitcast-convert does not support %s to %s.",
        primitive_util::LowercasePrimitiveTypeName(src_type),
        primitive_util::LowercasePrimitiveTypeName(dst_type)));
  }

  absl::StatusOr<std::string> EmitReduce(const HloInstruction* instr,
                                         std::vector<std::string>* body) {
    if (!IsPredArray(instr->shape()) || instr->operand_count() != 2 ||
        !IsPredArray(instr->operand(0)->shape()) ||
        instr->dimensions().size() != 1 || instr->dimensions()[0] != 1 ||
        instr->operand(0)->shape().dimensions().size() != 2 ||
        instr->operand(0)->shape().dimensions(1) != 1 ||
        instr->to_apply()->root_instruction()->opcode() != HloOpcode::kAnd ||
        ShapeUtil::ElementsIn(instr->shape()) !=
            ShapeUtil::ElementsIn(result_shape_)) {
      return absl::UnimplementedError(
          "Metal direct AIR reduce currently supports only pred rank-2 "
          "degenerate reduce-and to the result element count.");
    }
    return EmitValue(instr->operand(0), /*force_scalar=*/false, body);
  }

  absl::StatusOr<std::string> EmitSelect(const HloInstruction* instr,
                                         std::vector<std::string>* body) {
    if (instr->operand(1)->opcode() == HloOpcode::kGather) {
      auto pred_values = EvaluatePredArray(instr->operand(0));
      if (pred_values.ok() &&
          absl::c_all_of(*pred_values, [](bool value) { return value; })) {
        return EmitGather(instr->operand(1), body);
      }
    }
    if (!IsF32Array(instr->shape()) && !IsF16Array(instr->shape()) &&
        !IsBF16Array(instr->shape())) {
      if (!IsS32Array(instr->shape())) {
        return absl::UnimplementedError(
            "Metal direct AIR elementwise select currently supports only "
            "floating-point and s32 array results.");
      }
    }
    TF_ASSIGN_OR_RETURN(std::string pred,
                        EmitValue(instr->operand(0), /*force_scalar=*/false,
                                  body));
    TF_ASSIGN_OR_RETURN(std::string on_true,
                        EmitValue(instr->operand(1), IsScalarOperand(instr, 1),
                                  body));
    TF_ASSIGN_OR_RETURN(std::string on_false,
                        EmitValue(instr->operand(2), IsScalarOperand(instr, 2),
                                  body));
    std::string value = NewName("select");
    const char* ir_type = ValueIrType(instr->shape().element_type());
    body->push_back(absl::StrFormat("  %s = select i1 %s, %s %s, %s %s",
                                    value, pred, ir_type, on_true, ir_type,
                                    on_false));
    return value;
  }

  absl::StatusOr<std::string> EmitPad(const HloInstruction* instr,
                                      bool force_scalar,
                                      std::vector<std::string>* body) {
    const HloInstruction* input = instr->operand(0);
    const int64_t rank = instr->shape().dimensions().size();
    if (force_scalar || (!IsF32Array(instr->shape()) &&
                         !IsS32Array(instr->shape())) ||
        (rank != 1 && rank != 2) ||
        input->shape().dimensions().size() != rank ||
        instr->shape().element_type() != input->shape().element_type() ||
        instr->padding_config().dimensions_size() != rank ||
        ShapeUtil::ElementsIn(instr->shape()) !=
            ShapeUtil::ElementsIn(result_shape_)) {
      return absl::UnimplementedError(
          "Metal direct AIR elementwise pad currently supports only rank-1/rank-2 "
          "f32/s32 array results.");
    }
    for (int64_t dim = 0; dim < rank; ++dim) {
      const auto& dimension = instr->padding_config().dimensions(dim);
      if (dimension.edge_padding_low() < 0 ||
          dimension.edge_padding_high() < 0 ||
          dimension.interior_padding() != 0 ||
          input->shape().dimensions(dim) <= 0 ||
          input->shape().dimensions(dim) + dimension.edge_padding_low() +
                  dimension.edge_padding_high() !=
              instr->shape().dimensions(dim)) {
        return absl::UnimplementedError(
            "Metal direct AIR elementwise pad currently supports only "
            "non-negative edge padding without interior padding.");
      }
    }

    TF_ASSIGN_OR_RETURN(std::string pad_value,
                        EmitValue(instr->operand(1), /*force_scalar=*/true,
                                  body));
    std::string in_input;
    std::string safe_index;
    if (rank == 1) {
      const auto& dimension = instr->padding_config().dimensions(0);
      const int64_t low_padding = dimension.edge_padding_low();
      const int64_t input_elements = ShapeUtil::ElementsIn(input->shape());
      std::string after_low = NewName("pad_after_low");
      std::string before_end = NewName("pad_before_end");
      in_input = NewName("pad_in_input");
      body->push_back(absl::StrFormat("  %s = icmp uge i64 %%idx, %d",
                                      after_low, low_padding));
      body->push_back(absl::StrFormat("  %s = icmp ult i64 %%idx, %d",
                                      before_end,
                                      low_padding + input_elements));
      body->push_back(absl::StrFormat("  %s = and i1 %s, %s", in_input,
                                      after_low, before_end));

      std::string source_index = "%idx";
      if (low_padding != 0) {
        source_index = NewName("pad_source");
        body->push_back(absl::StrFormat("  %s = sub i64 %%idx, %d",
                                        source_index, low_padding));
      }
      std::string low_clamped = NewName("pad_low_clamped");
      safe_index = NewName("pad_safe_index");
      body->push_back(absl::StrFormat(
          "  %s = select i1 %s, i64 %s, i64 0", low_clamped, after_low,
          source_index));
      body->push_back(absl::StrFormat(
          "  %s = select i1 %s, i64 %s, i64 %d", safe_index, before_end,
          low_clamped, input_elements - 1));
    } else {
      const int64_t result_cols = instr->shape().dimensions(1);
      const int64_t input_cols = input->shape().dimensions(1);
      const int64_t row_low =
          instr->padding_config().dimensions(0).edge_padding_low();
      const int64_t col_low =
          instr->padding_config().dimensions(1).edge_padding_low();
      std::string row = NewName("pad_row");
      std::string col = NewName("pad_col");
      body->push_back(
          absl::StrFormat("  %s = udiv i64 %%idx, %d", row, result_cols));
      body->push_back(
          absl::StrFormat("  %s = urem i64 %%idx, %d", col, result_cols));
      std::string row_after_low = NewName("pad_row_after_low");
      std::string row_before_end = NewName("pad_row_before_end");
      std::string col_after_low = NewName("pad_col_after_low");
      std::string col_before_end = NewName("pad_col_before_end");
      std::string row_in = NewName("pad_row_in");
      std::string col_in = NewName("pad_col_in");
      in_input = NewName("pad_in_input");
      body->push_back(absl::StrFormat("  %s = icmp uge i64 %s, %d",
                                      row_after_low, row, row_low));
      body->push_back(absl::StrFormat("  %s = icmp ult i64 %s, %d",
                                      row_before_end, row,
                                      row_low + input->shape().dimensions(0)));
      body->push_back(absl::StrFormat("  %s = icmp uge i64 %s, %d",
                                      col_after_low, col, col_low));
      body->push_back(absl::StrFormat("  %s = icmp ult i64 %s, %d",
                                      col_before_end, col,
                                      col_low + input->shape().dimensions(1)));
      body->push_back(absl::StrFormat("  %s = and i1 %s, %s", row_in,
                                      row_after_low, row_before_end));
      body->push_back(absl::StrFormat("  %s = and i1 %s, %s", col_in,
                                      col_after_low, col_before_end));
      body->push_back(absl::StrFormat("  %s = and i1 %s, %s", in_input,
                                      row_in, col_in));

      std::string source_row = row;
      if (row_low != 0) {
        source_row = NewName("pad_source_row");
        body->push_back(absl::StrFormat("  %s = sub i64 %s, %d", source_row,
                                        row, row_low));
      }
      std::string source_col = col;
      if (col_low != 0) {
        source_col = NewName("pad_source_col");
        body->push_back(absl::StrFormat("  %s = sub i64 %s, %d", source_col,
                                        col, col_low));
      }
      std::string safe_row = NewName("pad_safe_row");
      std::string safe_col = NewName("pad_safe_col");
      std::string row_offset = NewName("pad_row_offset");
      safe_index = NewName("pad_safe_index");
      body->push_back(absl::StrFormat(
          "  %s = select i1 %s, i64 %s, i64 0", safe_row, row_in,
          source_row));
      body->push_back(absl::StrFormat(
          "  %s = select i1 %s, i64 %s, i64 0", safe_col, col_in,
          source_col));
      body->push_back(absl::StrFormat("  %s = mul i64 %s, %d", row_offset,
                                      safe_row, input_cols));
      body->push_back(absl::StrFormat("  %s = add i64 %s, %s", safe_index,
                                      row_offset, safe_col));
    }
    TF_ASSIGN_OR_RETURN(std::string input_value,
                        EmitLoadFromLinearIndex(input, safe_index, body));
    return EmitTypedSelect(instr->shape().element_type(), in_input, input_value,
                           pad_value, body, "pad_select");
  }

  absl::StatusOr<std::string> EmitLogicalBinary(
      const HloInstruction* instr, std::vector<std::string>* body) {
    if (instr->shape().element_type() == S32) {
      TF_ASSIGN_OR_RETURN(
          std::string lhs,
          EmitValue(instr->operand(0), IsScalarOperand(instr, 0), body));
      TF_ASSIGN_OR_RETURN(
          std::string rhs,
          EmitValue(instr->operand(1), IsScalarOperand(instr, 1), body));
      switch (instr->opcode()) {
        case HloOpcode::kAnd:
          return EmitOp("and i32", lhs, rhs, body);
        case HloOpcode::kOr:
          return EmitOp("or i32", lhs, rhs, body);
        case HloOpcode::kXor:
          return EmitOp("xor i32", lhs, rhs, body);
        default:
          return absl::InternalError("Unexpected s32 logical HLO opcode.");
      }
    }
    if (!IsPredArray(instr->shape())) {
      return absl::UnimplementedError(
          "Metal direct AIR logical binary ops currently support only pred "
          "and s32 arrays.");
    }
    TF_ASSIGN_OR_RETURN(std::string lhs,
                        EmitValue(instr->operand(0), IsScalarOperand(instr, 0),
                                  body));
    TF_ASSIGN_OR_RETURN(std::string rhs,
                        EmitValue(instr->operand(1), IsScalarOperand(instr, 1),
                                  body));
    std::string value = NewName("logical");
    switch (instr->opcode()) {
      case HloOpcode::kAnd:
        body->push_back(
            absl::StrFormat("  %s = and i1 %s, %s", value, lhs, rhs));
        break;
      case HloOpcode::kOr:
        body->push_back(
            absl::StrFormat("  %s = or i1 %s, %s", value, lhs, rhs));
        break;
      case HloOpcode::kXor:
        body->push_back(
            absl::StrFormat("  %s = xor i1 %s, %s", value, lhs, rhs));
        break;
      default:
        return absl::InternalError("Unexpected pred logical HLO opcode.");
    }
    return value;
  }

  absl::StatusOr<std::string> EmitShift(const HloInstruction* instr,
                                        std::vector<std::string>* body) {
    if (!IsS32Array(instr->shape())) {
      return absl::UnimplementedError(
          "Metal direct AIR shift ops currently support only s32 arrays.");
    }
    TF_ASSIGN_OR_RETURN(std::string lhs,
                        EmitValue(instr->operand(0), IsScalarOperand(instr, 0),
                                  body));
    TF_ASSIGN_OR_RETURN(std::string rhs,
                        EmitValue(instr->operand(1), IsScalarOperand(instr, 1),
                                  body));
    std::string in_range = NewName("shift_in_range");
    std::string safe_rhs = NewName("shift_safe_rhs");
    std::string shifted = NewName("shifted");
    std::string saturated = "0";
    body->push_back(
        absl::StrFormat("  %s = icmp ult i32 %s, 32", in_range, rhs));
    body->push_back(absl::StrFormat(
        "  %s = select i1 %s, i32 %s, i32 0", safe_rhs, in_range, rhs));
    switch (instr->opcode()) {
      case HloOpcode::kShiftLeft:
        body->push_back(
            absl::StrFormat("  %s = shl i32 %s, %s", shifted, lhs, safe_rhs));
        break;
      case HloOpcode::kShiftRightArithmetic: {
        body->push_back(absl::StrFormat("  %s = ashr i32 %s, %s", shifted,
                                        lhs, safe_rhs));
        std::string negative = NewName("shift_negative");
        std::string sign_bits = NewName("shift_sign_bits");
        body->push_back(absl::StrFormat("  %s = icmp slt i32 %s, 0",
                                        negative, lhs));
        body->push_back(absl::StrFormat(
            "  %s = select i1 %s, i32 -1, i32 0", sign_bits, negative));
        saturated = sign_bits;
        break;
      }
      case HloOpcode::kShiftRightLogical:
        body->push_back(
            absl::StrFormat("  %s = lshr i32 %s, %s", shifted, lhs, safe_rhs));
        break;
      default:
        return absl::InternalError("Unexpected shift HLO opcode.");
    }
    std::string result = NewName("shift_result");
    body->push_back(absl::StrFormat(
        "  %s = select i1 %s, i32 %s, i32 %s", result, in_range, shifted,
        saturated));
    return result;
  }

  absl::StatusOr<std::string> EmitNot(const HloInstruction* instr,
                                      std::vector<std::string>* body) {
    TF_ASSIGN_OR_RETURN(std::string value,
                        EmitValue(instr->operand(0), IsScalarOperand(instr, 0),
                                  body));
    if (instr->shape().element_type() == S32) {
      return EmitOp("xor i32", value, "-1", body);
    }
    if (instr->shape().element_type() == PRED) {
      return EmitOp("xor i1", value, "true", body);
    }
    return absl::UnimplementedError(
        "Metal direct AIR not currently supports only s32 and pred arrays.");
  }

  absl::StatusOr<std::string> EmitPopcount(const HloInstruction* instr,
                                           std::vector<std::string>* body) {
    if (!IsS32Array(instr->shape())) {
      return absl::UnimplementedError(
          "Metal direct AIR population count currently supports only s32 "
          "arrays.");
    }
    TF_ASSIGN_OR_RETURN(std::string value,
                        EmitValue(instr->operand(0), IsScalarOperand(instr, 0),
                                  body));
    std::string shifted1 = NewName("pop_shift1");
    std::string masked1 = NewName("pop_mask1");
    std::string partial1 = NewName("pop_partial1");
    std::string masked2_lhs = NewName("pop_mask2_lhs");
    std::string shifted2 = NewName("pop_shift2");
    std::string masked2_rhs = NewName("pop_mask2_rhs");
    std::string partial2 = NewName("pop_partial2");
    std::string shifted4 = NewName("pop_shift4");
    std::string partial4 = NewName("pop_partial4");
    std::string masked4 = NewName("pop_mask4");
    std::string product = NewName("pop_product");
    std::string result = NewName("pop_result");
    body->push_back(
        absl::StrFormat("  %s = lshr i32 %s, 1", shifted1, value));
    body->push_back(absl::StrFormat("  %s = and i32 %s, 1431655765",
                                    masked1, shifted1));
    body->push_back(absl::StrFormat("  %s = sub i32 %s, %s", partial1,
                                    value, masked1));
    body->push_back(absl::StrFormat("  %s = and i32 %s, 858993459",
                                    masked2_lhs, partial1));
    body->push_back(
        absl::StrFormat("  %s = lshr i32 %s, 2", shifted2, partial1));
    body->push_back(absl::StrFormat("  %s = and i32 %s, 858993459",
                                    masked2_rhs, shifted2));
    body->push_back(absl::StrFormat("  %s = add i32 %s, %s", partial2,
                                    masked2_lhs, masked2_rhs));
    body->push_back(
        absl::StrFormat("  %s = lshr i32 %s, 4", shifted4, partial2));
    body->push_back(absl::StrFormat("  %s = add i32 %s, %s", partial4,
                                    partial2, shifted4));
    body->push_back(absl::StrFormat("  %s = and i32 %s, 252645135", masked4,
                                    partial4));
    body->push_back(absl::StrFormat("  %s = mul i32 %s, 16843009", product,
                                    masked4));
    body->push_back(
        absl::StrFormat("  %s = lshr i32 %s, 24", result, product));
    return result;
  }

  absl::StatusOr<std::string> EmitGather(const HloInstruction* instr,
                                         std::vector<std::string>* body) {
    const GatherDimensionNumbers& dnums = instr->gather_dimension_numbers();
    if (IsF32Array(instr->shape()) && IsF32Array(instr->operand(0)->shape()) &&
        instr->shape().dimensions().size() == 2 &&
        instr->operand(0)->shape().dimensions().size() == 2) {
      return EmitRank2Gather(instr, body);
    }
    if (!IsF32Array(instr->shape()) || !IsF32Array(instr->operand(0)->shape()) ||
        instr->shape().dimensions().size() != 1 ||
        instr->operand(0)->shape().dimensions().size() != 1 ||
        instr->operand(1)->shape().dimensions().size() != 2 ||
        instr->operand(1)->shape().dimensions(1) != 1 ||
        ShapeUtil::ElementsIn(instr->shape()) !=
            instr->operand(1)->shape().dimensions(0) ||
        dnums.offset_dims_size() != 0 ||
        dnums.collapsed_slice_dims_size() != 1 ||
        dnums.collapsed_slice_dims(0) != 0 ||
        dnums.start_index_map_size() != 1 || dnums.start_index_map(0) != 0 ||
        dnums.index_vector_dim() != 1 ||
        instr->gather_slice_sizes().size() != 1 ||
        instr->gather_slice_sizes()[0] != 1) {
      return absl::UnimplementedError(
          "Metal direct AIR gather currently supports only rank-1 f32 gathers "
          "with scalar start indices.");
    }

    const int64_t input_elements =
        ShapeUtil::ElementsIn(instr->operand(0)->shape());
    auto indices_or = EvaluateS32Array(instr->operand(1));
    if (indices_or.ok()) {
      const std::vector<int64_t>& indices = *indices_or;
      if (indices.empty() ||
          indices.size() != ShapeUtil::ElementsIn(instr->shape())) {
        return absl::InvalidArgumentError(
            "Metal direct AIR gather index count does not match output size.");
      }
      for (int64_t index : indices) {
        if (index < 0 || index >= input_elements) {
          return absl::UnimplementedError(
              "Metal direct AIR gather currently requires static in-bounds "
              "indices.");
        }
      }

      std::string source_index = absl::StrCat(indices[0]);
      for (int64_t i = 1; i < indices.size(); ++i) {
        std::string is_lane = NewName("gather_lane");
        std::string selected = NewName("gather_index");
        body->push_back(
            absl::StrFormat("  %s = icmp eq i64 %%idx, %d", is_lane, i));
        body->push_back(absl::StrFormat(
            "  %s = select i1 %s, i64 %d, i64 %s", selected, is_lane,
            indices[i], source_index));
        source_index = selected;
      }
      return EmitLoadFromLinearIndex(instr->operand(0), source_index, body);
    }

    TF_ASSIGN_OR_RETURN(std::string raw_index,
                        EmitValue(instr->operand(1), /*force_scalar=*/false,
                                  body));
    std::string below_zero = NewName("gather_below_zero");
    std::string above_end = NewName("gather_above_end");
    std::string low_clamped = NewName("gather_low_clamped");
    std::string clamped = NewName("gather_clamped");
    std::string source_index = NewName("gather_source_index");
    body->push_back(absl::StrFormat("  %s = icmp slt i32 %s, 0", below_zero,
                                    raw_index));
    body->push_back(absl::StrFormat("  %s = icmp sge i32 %s, %d", above_end,
                                    raw_index, input_elements));
    body->push_back(absl::StrFormat(
        "  %s = select i1 %s, i32 0, i32 %s", low_clamped, below_zero,
        raw_index));
    body->push_back(absl::StrFormat(
        "  %s = select i1 %s, i32 %d, i32 %s", clamped, above_end,
        input_elements - 1, low_clamped));
    body->push_back(absl::StrFormat("  %s = sext i32 %s to i64",
                                    source_index, clamped));
    return EmitLoadFromLinearIndex(instr->operand(0), source_index, body);
  }

  absl::StatusOr<std::string> EmitRank2Gather(
      const HloInstruction* instr, std::vector<std::string>* body) {
    const HloInstruction* input = instr->operand(0);
    const HloInstruction* indices = instr->operand(1);
    const GatherDimensionNumbers& dnums = instr->gather_dimension_numbers();
    if (!IsS32Array(indices->shape()) ||
        indices->shape().dimensions().size() != 2 ||
        indices->shape().dimensions(1) != 1 ||
        dnums.offset_dims_size() != 1 ||
        dnums.collapsed_slice_dims_size() != 1 ||
        dnums.start_index_map_size() != 1 || dnums.index_vector_dim() != 1 ||
        instr->gather_slice_sizes().size() != 2) {
      return absl::UnimplementedError(
          "Metal direct AIR rank-2 gather requires rank-2 scalar start "
          "indices.");
    }

    const int64_t input_rows = input->shape().dimensions(0);
    const int64_t input_cols = input->shape().dimensions(1);
    const int64_t index_count = indices->shape().dimensions(0);
    const bool gather_rows =
        dnums.offset_dims(0) == 1 && dnums.collapsed_slice_dims(0) == 0 &&
        dnums.start_index_map(0) == 0 && instr->gather_slice_sizes()[0] == 1 &&
        instr->gather_slice_sizes()[1] == input_cols &&
        instr->shape().dimensions(0) == index_count &&
        instr->shape().dimensions(1) == input_cols;
    const bool gather_cols =
        dnums.offset_dims(0) == 0 && dnums.collapsed_slice_dims(0) == 1 &&
        dnums.start_index_map(0) == 1 &&
        instr->gather_slice_sizes()[0] == input_rows &&
        instr->gather_slice_sizes()[1] == 1 &&
        instr->shape().dimensions(0) == input_rows &&
        instr->shape().dimensions(1) == index_count;
    if (!gather_rows && !gather_cols) {
      return absl::UnimplementedError(
          "Metal direct AIR rank-2 gather currently supports gathering rows "
          "or columns with scalar start indices.");
    }

    const int64_t result_cols = instr->shape().dimensions(1);
    std::string row = NewName("gather_row");
    std::string col = NewName("gather_col");
    body->push_back(
        absl::StrFormat("  %s = udiv i64 %%idx, %d", row, result_cols));
    body->push_back(
        absl::StrFormat("  %s = urem i64 %%idx, %d", col, result_cols));

    TF_ASSIGN_OR_RETURN(
        std::string raw_index,
        EmitLoadFromLinearIndex(indices, gather_rows ? row : col, body));
    std::string clamped_index = EmitClampedStartIndex(
        raw_index, gather_rows ? input_rows - 1 : input_cols - 1, body);

    std::string source_index;
    if (gather_rows) {
      std::string row_offset = NewName("gather_row_offset");
      source_index = NewName("gather_source_index");
      body->push_back(absl::StrFormat("  %s = mul i64 %s, %d", row_offset,
                                      clamped_index, input_cols));
      body->push_back(absl::StrFormat("  %s = add i64 %s, %s", source_index,
                                      row_offset, col));
    } else {
      std::string row_offset = NewName("gather_row_offset");
      source_index = NewName("gather_source_index");
      body->push_back(
          absl::StrFormat("  %s = mul i64 %s, %d", row_offset, row, input_cols));
      body->push_back(absl::StrFormat("  %s = add i64 %s, %s", source_index,
                                      row_offset, clamped_index));
    }
    return EmitLoadFromLinearIndex(input, source_index, body);
  }

  absl::StatusOr<std::vector<int64_t>> EvaluateS32Array(
      const HloInstruction* instr) {
    if (instr->opcode() == HloOpcode::kParameter) {
      if (const HloInstruction* override = CallParameterOverride(instr)) {
        return EvaluateS32Array(override);
      }
    }
    if (instr->opcode() == HloOpcode::kBitcast ||
        instr->opcode() == HloOpcode::kReshape) {
      TF_ASSIGN_OR_RETURN(std::vector<int64_t> values,
                          EvaluateS32Array(instr->operand(0)));
      if (values.size() != ShapeUtil::ElementsIn(instr->shape())) {
        return absl::InvalidArgumentError(
            "Metal direct AIR constant s32 reshape changes element count.");
      }
      return values;
    }
    if (instr->opcode() == HloOpcode::kBroadcast &&
        instr->operand(0)->IsConstant() &&
        IsScalarLikeS32(instr->operand(0)->shape())) {
      return std::vector<int64_t>(
          ShapeUtil::ElementsIn(instr->shape()),
          instr->operand(0)->literal().Get<int32_t>({}));
    }
    if (instr->IsConstant() && instr->shape().element_type() == S32) {
      std::vector<int64_t> values;
      values.reserve(ShapeUtil::ElementsIn(instr->shape()));
      if (ShapeUtil::IsEffectiveScalar(instr->shape())) {
        values.push_back(instr->literal().Get<int32_t>({}));
        return values;
      }
      if (instr->shape().dimensions().size() == 1) {
        for (int64_t i = 0; i < instr->shape().dimensions(0); ++i) {
          values.push_back(instr->literal().Get<int32_t>({i}));
        }
        return values;
      }
      if (instr->shape().dimensions().size() == 2) {
        for (int64_t row = 0; row < instr->shape().dimensions(0); ++row) {
          for (int64_t col = 0; col < instr->shape().dimensions(1); ++col) {
            values.push_back(instr->literal().Get<int32_t>({row, col}));
          }
        }
        return values;
      }
    }
    if (instr->opcode() == HloOpcode::kCall) {
      const HloComputation* callee = instr->to_apply();
      if (callee->num_parameters() != instr->operand_count()) {
        return absl::InvalidArgumentError(
            "Metal direct AIR constant call operand count mismatch.");
      }
      CallParameterScope scope;
      scope.computation = callee;
      for (int64_t i = 0; i < instr->operand_count(); ++i) {
        scope.arguments[i] = instr->operand(i);
      }
      call_parameter_scopes_.push_back(std::move(scope));
      absl::Cleanup pop_scope = [this] { call_parameter_scopes_.pop_back(); };
      return EvaluateS32Array(callee->root_instruction());
    }
    if (instr->opcode() == HloOpcode::kAdd ||
        instr->opcode() == HloOpcode::kSubtract) {
      TF_ASSIGN_OR_RETURN(std::vector<int64_t> lhs,
                          EvaluateS32Array(instr->operand(0)));
      TF_ASSIGN_OR_RETURN(std::vector<int64_t> rhs,
                          EvaluateS32Array(instr->operand(1)));
      if (lhs.size() != rhs.size()) {
        return absl::InvalidArgumentError(
            "Metal direct AIR constant s32 binary size mismatch.");
      }
      for (int64_t i = 0; i < lhs.size(); ++i) {
        lhs[i] = instr->opcode() == HloOpcode::kAdd ? lhs[i] + rhs[i]
                                                    : lhs[i] - rhs[i];
      }
      return lhs;
    }
    if (instr->opcode() == HloOpcode::kSelect) {
      TF_ASSIGN_OR_RETURN(std::vector<bool> pred,
                          EvaluatePredArray(instr->operand(0)));
      TF_ASSIGN_OR_RETURN(std::vector<int64_t> on_true,
                          EvaluateS32Array(instr->operand(1)));
      TF_ASSIGN_OR_RETURN(std::vector<int64_t> on_false,
                          EvaluateS32Array(instr->operand(2)));
      if (pred.size() != on_true.size() || pred.size() != on_false.size()) {
        return absl::InvalidArgumentError(
            "Metal direct AIR constant s32 select size mismatch.");
      }
      for (int64_t i = 0; i < pred.size(); ++i) {
        on_true[i] = pred[i] ? on_true[i] : on_false[i];
      }
      return on_true;
    }
    return absl::UnimplementedError(absl::StrFormat(
        "Metal direct AIR cannot statically evaluate s32 HLO opcode %s.",
        HloOpcodeString(instr->opcode())));
  }

  absl::StatusOr<std::vector<bool>> EvaluatePredArray(
      const HloInstruction* instr) {
    if (instr->opcode() == HloOpcode::kParameter) {
      if (const HloInstruction* override = CallParameterOverride(instr)) {
        return EvaluatePredArray(override);
      }
    }
    if (instr->opcode() == HloOpcode::kBitcast ||
        instr->opcode() == HloOpcode::kReshape) {
      TF_ASSIGN_OR_RETURN(std::vector<bool> values,
                          EvaluatePredArray(instr->operand(0)));
      if (values.size() != ShapeUtil::ElementsIn(instr->shape())) {
        return absl::InvalidArgumentError(
            "Metal direct AIR constant pred reshape changes element count.");
      }
      return values;
    }
    if (instr->opcode() == HloOpcode::kBroadcast &&
        instr->operand(0)->IsConstant() &&
        instr->operand(0)->shape().element_type() == PRED) {
      return std::vector<bool>(
          ShapeUtil::ElementsIn(instr->shape()),
          instr->operand(0)->literal().Get<bool>({}));
    }
    if (instr->IsConstant() && instr->shape().element_type() == PRED) {
      std::vector<bool> values;
      values.reserve(ShapeUtil::ElementsIn(instr->shape()));
      if (ShapeUtil::IsEffectiveScalar(instr->shape())) {
        values.push_back(instr->literal().Get<bool>({}));
        return values;
      }
      if (instr->shape().dimensions().size() == 1) {
        for (int64_t i = 0; i < instr->shape().dimensions(0); ++i) {
          values.push_back(instr->literal().Get<bool>({i}));
        }
        return values;
      }
      if (instr->shape().dimensions().size() == 2) {
        for (int64_t row = 0; row < instr->shape().dimensions(0); ++row) {
          for (int64_t col = 0; col < instr->shape().dimensions(1); ++col) {
            values.push_back(instr->literal().Get<bool>({row, col}));
          }
        }
        return values;
      }
    }
    if (instr->opcode() == HloOpcode::kCall) {
      const HloComputation* callee = instr->to_apply();
      if (callee->num_parameters() != instr->operand_count()) {
        return absl::InvalidArgumentError(
            "Metal direct AIR constant call operand count mismatch.");
      }
      CallParameterScope scope;
      scope.computation = callee;
      for (int64_t i = 0; i < instr->operand_count(); ++i) {
        scope.arguments[i] = instr->operand(i);
      }
      call_parameter_scopes_.push_back(std::move(scope));
      absl::Cleanup pop_scope = [this] { call_parameter_scopes_.pop_back(); };
      return EvaluatePredArray(callee->root_instruction());
    }
    if (instr->opcode() == HloOpcode::kCompare) {
      TF_ASSIGN_OR_RETURN(std::vector<int64_t> lhs,
                          EvaluateS32Array(instr->operand(0)));
      TF_ASSIGN_OR_RETURN(std::vector<int64_t> rhs,
                          EvaluateS32Array(instr->operand(1)));
      if (lhs.size() != rhs.size()) {
        return absl::InvalidArgumentError(
            "Metal direct AIR constant compare size mismatch.");
      }
      std::vector<bool> values(lhs.size());
      for (int64_t i = 0; i < lhs.size(); ++i) {
        switch (instr->comparison_direction()) {
          case ComparisonDirection::kEq:
            values[i] = lhs[i] == rhs[i];
            break;
          case ComparisonDirection::kNe:
            values[i] = lhs[i] != rhs[i];
            break;
          case ComparisonDirection::kGe:
            values[i] = lhs[i] >= rhs[i];
            break;
          case ComparisonDirection::kGt:
            values[i] = lhs[i] > rhs[i];
            break;
          case ComparisonDirection::kLe:
            values[i] = lhs[i] <= rhs[i];
            break;
          case ComparisonDirection::kLt:
            values[i] = lhs[i] < rhs[i];
            break;
        }
      }
      return values;
    }
    if (instr->opcode() == HloOpcode::kAnd) {
      TF_ASSIGN_OR_RETURN(std::vector<bool> lhs,
                          EvaluatePredArray(instr->operand(0)));
      TF_ASSIGN_OR_RETURN(std::vector<bool> rhs,
                          EvaluatePredArray(instr->operand(1)));
      if (lhs.size() != rhs.size()) {
        return absl::InvalidArgumentError(
            "Metal direct AIR constant pred binary size mismatch.");
      }
      for (int64_t i = 0; i < lhs.size(); ++i) {
        lhs[i] = lhs[i] && rhs[i];
      }
      return lhs;
    }
    if (instr->opcode() == HloOpcode::kReduce &&
        instr->operand_count() == 2 &&
        instr->dimensions().size() == 1 && instr->dimensions()[0] == 1 &&
        instr->operand(0)->shape().dimensions().size() == 2 &&
        instr->to_apply()->root_instruction()->opcode() == HloOpcode::kAnd &&
        instr->operand(1)->IsConstant() &&
        instr->operand(1)->shape().element_type() == PRED) {
      TF_ASSIGN_OR_RETURN(std::vector<bool> input,
                          EvaluatePredArray(instr->operand(0)));
      const int64_t rows = instr->operand(0)->shape().dimensions(0);
      const int64_t cols = instr->operand(0)->shape().dimensions(1);
      std::vector<bool> values(rows,
                               instr->operand(1)->literal().Get<bool>({}));
      for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < cols; ++col) {
          values[row] = values[row] && input[row * cols + col];
        }
      }
      return values;
    }
    return absl::UnimplementedError(absl::StrFormat(
        "Metal direct AIR cannot statically evaluate pred HLO opcode %s.",
        HloOpcodeString(instr->opcode())));
  }

  absl::StatusOr<std::string> EmitBinary(const HloInstruction* instr,
                                         std::vector<std::string>* body) {
    TF_ASSIGN_OR_RETURN(std::string lhs,
                        EmitValue(instr->operand(0), IsScalarOperand(instr, 0),
                                  body));
    TF_ASSIGN_OR_RETURN(std::string rhs,
                        EmitValue(instr->operand(1), IsScalarOperand(instr, 1),
                                  body));
    const PrimitiveType type = instr->shape().element_type();
    if (IsIntegerElementType(type)) {
      const char* ir_type = ValueIrType(type);
      switch (instr->opcode()) {
        case HloOpcode::kAdd:
          return EmitOp(absl::StrCat("add ", ir_type), lhs, rhs, body);
        case HloOpcode::kSubtract:
          return EmitOp(absl::StrCat("sub ", ir_type), lhs, rhs, body);
        case HloOpcode::kMultiply:
          return EmitOp(absl::StrCat("mul ", ir_type), lhs, rhs, body);
        case HloOpcode::kDivide:
          if (type == S32) {
            return EmitOp("sdiv i32", lhs, rhs, body);
          }
          break;
        case HloOpcode::kRemainder:
          if (type == S32) {
            return EmitS32Remainder(lhs, rhs, body);
          }
          break;
        case HloOpcode::kMaximum:
          return EmitIntCompareSelect(
              IsSignedIntegerElementType(type) ? "sgt" : "ugt", lhs, rhs,
              body, type);
        case HloOpcode::kMinimum:
          return EmitIntCompareSelect(
              IsSignedIntegerElementType(type) ? "slt" : "ult", lhs, rhs,
              body, type);
        default:
          break;
      }
      return absl::UnimplementedError(
          "Metal direct AIR integer binary op is not supported for this "
          "element type.");
    }
    switch (instr->opcode()) {
      case HloOpcode::kAdd:
        return EmitOp("fadd fast float", lhs, rhs, body);
      case HloOpcode::kSubtract:
        return EmitOp("fsub fast float", lhs, rhs, body);
      case HloOpcode::kMultiply:
        return EmitOp("fmul fast float", lhs, rhs, body);
      case HloOpcode::kDivide:
        return EmitOp("fdiv fast float", lhs, rhs, body);
      case HloOpcode::kRemainder:
        return EmitF32Remainder(lhs, rhs, body);
      case HloOpcode::kMaximum:
        return EmitCompareSelect("ogt", lhs, rhs, body);
      case HloOpcode::kMinimum:
        return EmitCompareSelect("olt", lhs, rhs, body);
      default:
        return absl::InternalError("Unexpected binary HLO opcode.");
    }
  }

  absl::StatusOr<std::string> EmitF32ReducerOp(
      HloOpcode opcode, absl::string_view lhs, absl::string_view rhs,
      std::vector<std::string>* body) {
    switch (opcode) {
      case HloOpcode::kAdd:
        return EmitOp("fadd fast float", lhs, rhs, body);
      case HloOpcode::kMultiply:
        return EmitOp("fmul fast float", lhs, rhs, body);
      case HloOpcode::kMaximum:
        return EmitCompareSelect("ogt", lhs, rhs, body);
      case HloOpcode::kMinimum:
        return EmitCompareSelect("olt", lhs, rhs, body);
      default:
        return absl::InternalError("Unexpected f32 reducer HLO opcode.");
    }
  }

  absl::StatusOr<std::string> EmitS32Remainder(
      absl::string_view lhs, absl::string_view rhs,
      std::vector<std::string>* body) {
    std::string rhs_is_zero = NewName("rem_rhs_is_zero");
    std::string lhs_is_min = NewName("rem_lhs_is_min");
    std::string rhs_is_minus_one = NewName("rem_rhs_is_minus_one");
    std::string overflow = NewName("rem_overflow");
    std::string unsafe = NewName("rem_unsafe");
    std::string safe_rhs = NewName("rem_safe_rhs");
    std::string safe_rem = NewName("rem_safe");
    std::string overflow_result = NewName("rem_overflow_result");
    std::string result = NewName("rem_result");
    body->push_back(
        absl::StrFormat("  %s = icmp eq i32 %s, 0", rhs_is_zero, rhs));
    body->push_back(absl::StrFormat(
        "  %s = icmp eq i32 %s, -2147483648", lhs_is_min, lhs));
    body->push_back(absl::StrFormat(
        "  %s = icmp eq i32 %s, -1", rhs_is_minus_one, rhs));
    body->push_back(absl::StrFormat("  %s = and i1 %s, %s", overflow,
                                    lhs_is_min, rhs_is_minus_one));
    body->push_back(absl::StrFormat("  %s = or i1 %s, %s", unsafe,
                                    rhs_is_zero, overflow));
    body->push_back(absl::StrFormat(
        "  %s = select i1 %s, i32 1, i32 %s", safe_rhs, unsafe, rhs));
    body->push_back(
        absl::StrFormat("  %s = srem i32 %s, %s", safe_rem, lhs, safe_rhs));
    body->push_back(absl::StrFormat(
        "  %s = select i1 %s, i32 0, i32 %s", overflow_result, overflow,
        safe_rem));
    body->push_back(absl::StrFormat(
        "  %s = select i1 %s, i32 %s, i32 %s", result, rhs_is_zero, lhs,
        overflow_result));
    return result;
  }

  absl::StatusOr<std::string> EmitF32Remainder(
      absl::string_view lhs, absl::string_view rhs,
      std::vector<std::string>* body) {
    std::string raw = NewName("rem_raw");
    std::string same = NewName("rem_same");
    std::string rhs_zero = NewName("rem_rhs_zero");
    std::string rhs_nonzero = NewName("rem_rhs_nonzero");
    std::string force_zero = NewName("rem_force_zero");
    std::string result = NewName("rem_result");
    body->push_back(absl::StrFormat(
        "  %s = call fast float @air.fast_fmod.f32(float %s, float %s)", raw,
        lhs, rhs));
    body->push_back(
        absl::StrFormat("  %s = fcmp oeq float %s, %s", same, lhs, rhs));
    body->push_back(absl::StrFormat(
        "  %s = fcmp oeq float %s, 0x0000000000000000", rhs_zero, rhs));
    body->push_back(
        absl::StrFormat("  %s = xor i1 %s, true", rhs_nonzero, rhs_zero));
    body->push_back(absl::StrFormat("  %s = and i1 %s, %s", force_zero,
                                    same, rhs_nonzero));
    body->push_back(absl::StrFormat(
        "  %s = select i1 %s, float 0x0000000000000000, float %s", result,
        force_zero, raw));
    return result;
  }

  absl::StatusOr<std::string> EmitNegate(const HloInstruction* instr,
                                         std::vector<std::string>* body) {
    TF_ASSIGN_OR_RETURN(std::string value,
                        EmitValue(instr->operand(0), IsScalarOperand(instr, 0),
                                  body));
    std::string name = NewName("unary");
    if (instr->shape().element_type() == S32) {
      body->push_back(absl::StrFormat("  %s = sub i32 0, %s", name, value));
      return name;
    }
    if (IsFloatAccumulatorElementType(instr->shape().element_type())) {
      body->push_back(absl::StrFormat("  %s = fneg fast float %s", name,
                                      value));
      return name;
    }
    return absl::UnimplementedError(
        "Metal direct AIR negate currently supports only floating-point and "
        "s32 arrays.");
  }

  absl::StatusOr<std::string> EmitSign(const HloInstruction* instr,
                                       std::vector<std::string>* body) {
    if (instr->shape().element_type() == F32) {
      return EmitIntrinsicUnary(instr, "air.sign.f32", body);
    }
    if (instr->shape().element_type() != S32) {
      return absl::UnimplementedError(
          "Metal direct AIR sign currently supports only f32 and s32 arrays.");
    }
    TF_ASSIGN_OR_RETURN(std::string value,
                        EmitValue(instr->operand(0), IsScalarOperand(instr, 0),
                                  body));
    std::string positive = NewName("sign_positive");
    std::string negative = NewName("sign_negative");
    std::string nonnegative_result = NewName("sign_nonnegative_result");
    std::string result = NewName("sign_result");
    body->push_back(
        absl::StrFormat("  %s = icmp sgt i32 %s, 0", positive, value));
    body->push_back(
        absl::StrFormat("  %s = icmp slt i32 %s, 0", negative, value));
    body->push_back(absl::StrFormat(
        "  %s = select i1 %s, i32 1, i32 0", nonnegative_result, positive));
    body->push_back(absl::StrFormat(
        "  %s = select i1 %s, i32 -1, i32 %s", result, negative,
        nonnegative_result));
    return result;
  }

  absl::StatusOr<std::string> EmitErf(const HloInstruction* instr,
                                      std::vector<std::string>* body) {
    if (instr->shape().element_type() != F32) {
      return absl::UnimplementedError(
          "Metal direct AIR erf currently supports only f32 arrays.");
    }
    TF_ASSIGN_OR_RETURN(std::string value,
                        EmitValue(instr->operand(0), IsScalarOperand(instr, 0),
                                  body));
    std::string negative = NewName("erf_negative");
    std::string negated = NewName("erf_negated");
    std::string abs_value = NewName("erf_abs");
    body->push_back(absl::StrFormat(
        "  %s = fcmp fast olt float %s, 0x0000000000000000", negative, value));
    body->push_back(
        absl::StrFormat("  %s = fneg fast float %s", negated, value));
    body->push_back(absl::StrFormat(
        "  %s = select i1 %s, float %s, float %s", abs_value, negative,
        negated, value));

    std::string scaled = EmitOp("fmul fast float", FloatLiteral(0.3275911f),
                                abs_value, body);
    std::string denom =
        EmitOp("fadd fast float", "1.000000e+00", scaled, body);
    std::string t = EmitOp("fdiv fast float", "1.000000e+00", denom, body);

    std::string poly = EmitOp("fmul fast float", FloatLiteral(1.061405429f), t,
                              body);
    poly = EmitOp("fadd fast float", poly, FloatLiteral(-1.453152027f), body);
    poly = EmitOp("fmul fast float", poly, t, body);
    poly = EmitOp("fadd fast float", poly, FloatLiteral(1.421413741f), body);
    poly = EmitOp("fmul fast float", poly, t, body);
    poly = EmitOp("fadd fast float", poly, FloatLiteral(-0.284496736f), body);
    poly = EmitOp("fmul fast float", poly, t, body);
    poly = EmitOp("fadd fast float", poly, FloatLiteral(0.254829592f), body);
    poly = EmitOp("fmul fast float", poly, t, body);

    std::string squared =
        EmitOp("fmul fast float", abs_value, abs_value, body);
    std::string neg_squared = NewName("erf_neg_squared");
    body->push_back(absl::StrFormat("  %s = fneg fast float %s", neg_squared,
                                    squared));
    std::string exp = NewName("erf_exp");
    body->push_back(absl::StrFormat(
        "  %s = call fast float @air.fast_exp.f32(float %s)", exp,
        neg_squared));
    std::string tail = EmitOp("fmul fast float", poly, exp, body);
    std::string positive_result =
        EmitOp("fsub fast float", "1.000000e+00", tail, body);
    std::string negative_result = NewName("erf_negative_result");
    std::string result = NewName("erf_result");
    body->push_back(absl::StrFormat("  %s = fneg fast float %s",
                                    negative_result, positive_result));
    body->push_back(absl::StrFormat(
        "  %s = select i1 %s, float %s, float %s", result, negative,
        negative_result, positive_result));
    return result;
  }

  absl::StatusOr<std::string> EmitLog1p(const HloInstruction* instr,
                                        std::vector<std::string>* body) {
    TF_ASSIGN_OR_RETURN(std::string value,
                        EmitValue(instr->operand(0), IsScalarOperand(instr, 0),
                                  body));
    std::string one_plus =
        EmitOp("fadd fast float", value, "1.000000e+00", body);
    std::string name = NewName("log1p");
    body->push_back(absl::StrFormat(
        "  %s = call fast float @air.fast_log.f32(float %s)", name, one_plus));
    return name;
  }

  absl::StatusOr<std::string> EmitExpm1(const HloInstruction* instr,
                                        std::vector<std::string>* body) {
    TF_ASSIGN_OR_RETURN(std::string value,
                        EmitValue(instr->operand(0), IsScalarOperand(instr, 0),
                                  body));
    std::string exp = NewName("exp");
    body->push_back(absl::StrFormat(
        "  %s = call fast float @air.fast_exp.f32(float %s)", exp, value));
    return EmitOp("fsub fast float", exp, "1.000000e+00", body);
  }

  absl::StatusOr<std::string> EmitLogistic(const HloInstruction* instr,
                                           std::vector<std::string>* body) {
    TF_ASSIGN_OR_RETURN(std::string value,
                        EmitValue(instr->operand(0), IsScalarOperand(instr, 0),
                                  body));
    std::string neg = NewName("neg");
    body->push_back(absl::StrFormat("  %s = fneg fast float %s", neg, value));
    std::string exp = NewName("exp");
    body->push_back(absl::StrFormat(
        "  %s = call fast float @air.fast_exp.f32(float %s)", exp, neg));
    std::string denom =
        EmitOp("fadd fast float", "1.000000e+00", exp, body);
    return EmitOp("fdiv fast float", "1.000000e+00", denom, body);
  }

  absl::StatusOr<std::string> EmitIntrinsicUnary(
      const HloInstruction* instr, absl::string_view intrinsic,
      std::vector<std::string>* body) {
    TF_ASSIGN_OR_RETURN(std::string value,
                        EmitValue(instr->operand(0), IsScalarOperand(instr, 0),
                                  body));
    std::string name = NewName("intrinsic");
    body->push_back(absl::StrFormat("  %s = call fast float @%s(float %s)",
                                    name, intrinsic, value));
    return name;
  }

  absl::StatusOr<std::string> EmitIntrinsicBinary(
      const HloInstruction* instr, absl::string_view intrinsic,
      std::vector<std::string>* body) {
    if (!IsF32Array(instr->shape())) {
      return absl::UnimplementedError(
          "Metal direct AIR intrinsic binary ops currently support only f32 "
          "array results.");
    }
    TF_ASSIGN_OR_RETURN(std::string lhs,
                        EmitValue(instr->operand(0), IsScalarOperand(instr, 0),
                                  body));
    TF_ASSIGN_OR_RETURN(std::string rhs,
                        EmitValue(instr->operand(1), IsScalarOperand(instr, 1),
                                  body));
    std::string name = NewName("intrinsic");
    body->push_back(absl::StrFormat(
        "  %s = call fast float @%s(float %s, float %s)", name, intrinsic, lhs,
        rhs));
    return name;
  }

  absl::StatusOr<std::string> EmitAtan2(const HloInstruction* instr,
                                        std::vector<std::string>* body) {
    if (!IsF32Array(instr->shape())) {
      return absl::UnimplementedError(
          "Metal direct AIR atan2 currently supports only f32 array results.");
    }
    TF_ASSIGN_OR_RETURN(std::string lhs,
                        EmitValue(instr->operand(0), IsScalarOperand(instr, 0),
                                  body));
    TF_ASSIGN_OR_RETURN(std::string rhs,
                        EmitValue(instr->operand(1), IsScalarOperand(instr, 1),
                                  body));
    std::string value = NewName("atan2");
    body->push_back(absl::StrFormat(
        "  %s = call fast float @air.fast_atan2.f32(float %s, float %s)",
        value, lhs, rhs));
    std::string lhs_zero = NewName("atan2_lhs_zero");
    std::string rhs_zero = NewName("atan2_rhs_zero");
    std::string both_zero = NewName("atan2_both_zero");
    std::string selected = NewName("atan2_select");
    body->push_back(absl::StrFormat(
        "  %s = fcmp fast oeq float %s, 0x0000000000000000", lhs_zero, lhs));
    body->push_back(absl::StrFormat(
        "  %s = fcmp fast oeq float %s, 0x0000000000000000", rhs_zero, rhs));
    body->push_back(absl::StrFormat("  %s = and i1 %s, %s", both_zero,
                                    lhs_zero, rhs_zero));
    body->push_back(absl::StrFormat(
        "  %s = select i1 %s, float 0x0000000000000000, float %s", selected,
        both_zero, value));
    return selected;
  }

  absl::StatusOr<std::string> EmitAbs(const HloInstruction* instr,
                                      std::vector<std::string>* body) {
    TF_ASSIGN_OR_RETURN(std::string value,
                        EmitValue(instr->operand(0), IsScalarOperand(instr, 0),
                                  body));
    if (instr->shape().element_type() == S32) {
      std::string neg = NewName("neg");
      body->push_back(absl::StrFormat("  %s = sub i32 0, %s", neg, value));
      return EmitIntCompareSelect("sgt", value, neg, body);
    }
    if (!IsFloatAccumulatorElementType(instr->shape().element_type())) {
      return absl::UnimplementedError(
          "Metal direct AIR abs currently supports only floating-point and "
          "s32 arrays.");
    }
    std::string neg = NewName("neg");
    body->push_back(absl::StrFormat("  %s = fneg fast float %s", neg, value));
    return EmitCompareSelect("ogt", value, neg, body);
  }

  bool IsScalarOperand(const HloInstruction* instr, int operand_index) const {
    return IsScalarLikeSupported(instr->operand(operand_index)->shape());
  }

  bool HasResultDimensions(const Shape& shape) const {
    return shape.IsArray() && shape.dimensions() == result_shape_.dimensions();
  }

  std::string EmitLoad(int input_index, PrimitiveType type,
                       absl::string_view index, std::vector<std::string>* body) {
    const char* ir_type = ElementIrType(type);
    std::string ptr = NewName("ptr");
    std::string value = NewName("value");
    body->push_back(absl::StrFormat(
        "  %s = getelementptr inbounds %s, %s addrspace(1)* %%arg%d, "
        "i64 %s",
        ptr, ir_type, ir_type, input_index, index));
    body->push_back(absl::StrFormat(
        "  %s = load %s, %s addrspace(1)* %s, align %d", value, ir_type,
        ir_type, ptr, ElementTypeSize(type)));
    if (type == F16) {
      std::string converted = NewName("f16_float");
      body->push_back(absl::StrFormat("  %s = fpext half %s to float",
                                      converted, value));
      return converted;
    }
    if (type == BF16) {
      std::string extended = NewName("bf16_zext");
      std::string shifted = NewName("bf16_shift");
      std::string converted = NewName("bf16_float");
      body->push_back(
          absl::StrFormat("  %s = zext i16 %s to i32", extended, value));
      body->push_back(
          absl::StrFormat("  %s = shl i32 %s, 16", shifted, extended));
      body->push_back(
          absl::StrFormat("  %s = bitcast i32 %s to float", converted,
                          shifted));
      return converted;
    }
    if (type == PRED) {
      std::string pred = NewName("pred");
      body->push_back(
          absl::StrFormat("  %s = icmp ne i8 %s, 0", pred, value));
      return pred;
    }
    return value;
  }

  std::string EmitOp(absl::string_view op, absl::string_view lhs,
                     absl::string_view rhs, std::vector<std::string>* body) {
    std::string name = NewName("op");
    body->push_back(absl::StrFormat("  %s = %s %s, %s", name, op, lhs, rhs));
    return name;
  }

  std::string EmitCompareSelect(absl::string_view predicate,
                                absl::string_view lhs, absl::string_view rhs,
                                std::vector<std::string>* body) {
    std::string cmp = NewName("cmp");
    std::string value = NewName("select");
    body->push_back(absl::StrFormat("  %s = fcmp fast %s float %s, %s", cmp,
                                    predicate, lhs, rhs));
    body->push_back(absl::StrFormat(
        "  %s = select i1 %s, float %s, float %s", value, cmp, lhs, rhs));
    return value;
  }

  std::string EmitIntCompareSelect(absl::string_view predicate,
                                   absl::string_view lhs, absl::string_view rhs,
                                   std::vector<std::string>* body,
                                   PrimitiveType type = S32) {
    std::string cmp = NewName("cmp");
    std::string value = NewName("select");
    const char* ir_type = ElementIrType(type);
    body->push_back(absl::StrFormat("  %s = icmp %s %s %s, %s", cmp,
                                    predicate, ir_type, lhs, rhs));
    body->push_back(absl::StrFormat(
        "  %s = select i1 %s, %s %s, %s %s", value, cmp, ir_type, lhs,
        ir_type, rhs));
    return value;
  }

  std::string EmitTypedSelect(PrimitiveType type, absl::string_view pred,
                              absl::string_view on_true,
                              absl::string_view on_false,
                              std::vector<std::string>* body,
                              absl::string_view prefix) {
    std::string value = NewName(prefix);
    const char* ir_type = ValueIrType(type);
    body->push_back(absl::StrFormat("  %s = select i1 %s, %s %s, %s %s",
                                    value, pred, ir_type, on_true, ir_type,
                                    on_false));
    return value;
  }

  absl::StatusOr<std::string> EmitStartIndexAsS32(
      const HloInstruction* start, std::vector<std::string>* body) {
    const PrimitiveType type = start->shape().element_type();
    TF_ASSIGN_OR_RETURN(std::string value,
                        EmitValue(start, /*force_scalar=*/true, body));
    if (type == S32 || type == U32) {
      return value;
    }
    if (type == S8 || type == S16 || type == U8 || type == U16) {
      std::string converted = NewName("start_index_s32");
      body->push_back(absl::StrFormat(
          "  %s = %s %s %s to i32", converted,
          IsSignedIntegerElementType(type) ? "sext" : "zext",
          ElementIrType(type), value));
      return converted;
    }
    return absl::UnimplementedError(
        "Metal direct AIR start index conversion requires an integer scalar.");
  }

  std::string EmitClampedStartIndex(absl::string_view raw_start,
                                    int64_t max_start,
                                    std::vector<std::string>* body) {
    if (max_start <= 0) {
      return "0";
    }
    std::string below_zero = NewName("start_below_zero");
    std::string above_end = NewName("start_above_end");
    std::string low_clamped = NewName("start_low_clamped");
    std::string clamped = NewName("start_clamped");
    std::string clamped_i64 = NewName("start_i64");
    body->push_back(absl::StrFormat("  %s = icmp slt i32 %s, 0", below_zero,
                                    raw_start));
    body->push_back(absl::StrFormat("  %s = icmp sgt i32 %s, %d", above_end,
                                    raw_start, max_start));
    body->push_back(absl::StrFormat(
        "  %s = select i1 %s, i32 0, i32 %s", low_clamped, below_zero,
        raw_start));
    body->push_back(absl::StrFormat(
        "  %s = select i1 %s, i32 %d, i32 %s", clamped, above_end, max_start,
        low_clamped));
    body->push_back(
        absl::StrFormat("  %s = sext i32 %s to i64", clamped_i64, clamped));
    return clamped_i64;
  }

  std::string NewName(absl::string_view prefix) {
    return absl::StrFormat("%%%s%d", prefix, next_value_id_++);
  }

  std::string NewLabel(absl::string_view prefix) {
    return absl::StrFormat("%s%d", prefix, next_value_id_++);
  }

  std::string BuildModule() const {
    const PrimitiveType result_type = result_shape_.element_type();
    const bool result_is_pred = result_type == PRED;
    std::vector<std::string> args;
    for (int i = 0; i < parameter_numbers_.size(); ++i) {
      const char* ir_type = ElementIrType(parameter_types_[i]);
      args.push_back(absl::StrFormat(
          "    %s addrspace(1)* nocapture noundef readonly "
          "\"air-buffer-no-alias\" %%arg%d",
          ir_type, i));
    }
    args.push_back(absl::StrFormat(
        "    %s addrspace(1)* nocapture noundef writeonly "
        "\"air-buffer-no-alias\" %%out",
        ElementIrType(result_type)));
    args.push_back(
        "    %struct.ElementwiseParams addrspace(2)* nocapture noundef "
        "readonly align 4 dereferenceable(16) \"air-buffer-no-alias\" %params");
    args.push_back("    <3 x i32> noundef %gid");

    std::vector<std::string> signature_types;
    signature_types.reserve(parameter_numbers_.size() + 3);
    for (PrimitiveType parameter_type : parameter_types_) {
      signature_types.push_back(
          absl::StrCat(ElementIrType(parameter_type), " addrspace(1)*"));
    }
    signature_types.push_back(
        absl::StrCat(ElementIrType(result_type), " addrspace(1)*"));
    signature_types.push_back("%struct.ElementwiseParams addrspace(2)*");
    signature_types.push_back("<3 x i32>");

    std::vector<std::string> metadata_args;
    for (int i = 0; i < parameter_numbers_.size(); ++i) {
      metadata_args.push_back(absl::StrFormat("!%d", 3 + i));
    }
    const int output_metadata = 3 + parameter_numbers_.size();
    const int params_metadata = output_metadata + 1;
    const int struct_metadata = params_metadata + 1;
    const int gid_metadata = struct_metadata + 1;
    metadata_args.push_back(absl::StrFormat("!%d", output_metadata));
    metadata_args.push_back(absl::StrFormat("!%d", params_metadata));
    metadata_args.push_back(absl::StrFormat("!%d", gid_metadata));
    const int module_flags_metadata = gid_metadata + 1;
    const int compile_options_metadata = module_flags_metadata + 8;
    const int ident_metadata = compile_options_metadata + 3;
    const int version_metadata = ident_metadata + 1;
    const int language_version_metadata = version_metadata + 1;
    const int source_file_name_metadata = language_version_metadata + 1;
    std::vector<std::string> module_flag_refs;
    module_flag_refs.reserve(8);
    for (int i = 0; i < 8; ++i) {
      module_flag_refs.push_back(
          absl::StrFormat("!%d", module_flags_metadata + i));
    }
    std::vector<std::string> compile_option_refs;
    compile_option_refs.reserve(3);
    for (int i = 0; i < 3; ++i) {
      compile_option_refs.push_back(
          absl::StrFormat("!%d", compile_options_metadata + i));
    }

    std::string store_result;
    if (raw_byte_result_) {
      store_result = absl::StrFormat(R"(  %%out_ptr = getelementptr inbounds i8, i8 addrspace(1)* %%out, i64 %%idx
  store i8 %s, i8 addrspace(1)* %%out_ptr, align 1)",
                                     result_value_);
    } else if (result_is_pred) {
      store_result = absl::StrFormat(R"(  %%out_ptr = getelementptr inbounds i8, i8 addrspace(1)* %%out, i64 %%idx
  %%out_i8 = zext i1 %s to i8
  store i8 %%out_i8, i8 addrspace(1)* %%out_ptr, align 1)",
                                     result_value_);
    } else if (result_type == F16) {
      store_result = absl::StrFormat(R"(  %%out_ptr = getelementptr inbounds half, half addrspace(1)* %%out, i64 %%idx
  %%out_f16 = fptrunc float %s to half
  store half %%out_f16, half addrspace(1)* %%out_ptr, align 2)",
                                     result_value_);
    } else if (raw_bf16_result_) {
      store_result = absl::StrFormat(R"(  %%out_ptr = getelementptr inbounds i16, i16 addrspace(1)* %%out, i64 %%idx
  store i16 %s, i16 addrspace(1)* %%out_ptr, align 2)",
                                     result_value_);
    } else if (result_type == BF16) {
      store_result = absl::StrFormat(R"(  %%out_ptr = getelementptr inbounds i16, i16 addrspace(1)* %%out, i64 %%idx
  %%out_bits = bitcast float %s to i32
  %%out_lsb = lshr i32 %%out_bits, 16
  %%out_lsb1 = and i32 %%out_lsb, 1
  %%out_bias = add i32 32767, %%out_lsb1
  %%out_rounded = add i32 %%out_bits, %%out_bias
  %%out_bf32 = lshr i32 %%out_rounded, 16
  %%out_bf16 = trunc i32 %%out_bf32 to i16
  store i16 %%out_bf16, i16 addrspace(1)* %%out_ptr, align 2)",
                                     result_value_);
    } else if (result_type == S32) {
      store_result = absl::StrFormat(R"(  %%out_ptr = getelementptr inbounds i32, i32 addrspace(1)* %%out, i64 %%idx
  store i32 %s, i32 addrspace(1)* %%out_ptr, align 4)",
                                     result_value_);
    } else {
      const char* ir_type = ElementIrType(result_type);
      const int alignment = ElementTypeSize(result_type);
      store_result = absl::StrFormat(
          R"(  %%out_ptr = getelementptr inbounds %s, %s addrspace(1)* %%out, i64 %%idx
  store %s %s, %s addrspace(1)* %%out_ptr, align %d)",
          ir_type, ir_type, ir_type, result_value_, ir_type, alignment);
    }

    std::string module = absl::StrFormat(R"(
source_filename = "xla/service/gpu/metal_elementwise_air"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024-n8:16:32"
target triple = "air64_v27-apple-macosx15.0.0"

%%struct.ElementwiseParams = type { i32, i32, i32, i32 }

declare float @air.fast_cos.f32(float) local_unnamed_addr #1
declare float @air.fast_acos.f32(float) local_unnamed_addr #1
declare float @air.fast_acosh.f32(float) local_unnamed_addr #1
declare float @air.fast_atan2.f32(float, float) local_unnamed_addr #1
declare float @air.fast_asin.f32(float) local_unnamed_addr #1
declare float @air.fast_asinh.f32(float) local_unnamed_addr #1
declare float @air.fast_atanh.f32(float) local_unnamed_addr #1
declare float @air.fast_ceil.f32(float) local_unnamed_addr #1
declare float @air.fast_cosh.f32(float) local_unnamed_addr #1
declare float @air.fast_exp.f32(float) local_unnamed_addr #1
declare float @air.fast_floor.f32(float) local_unnamed_addr #1
declare float @air.fast_fmod.f32(float, float) local_unnamed_addr #1
declare float @air.fast_log.f32(float) local_unnamed_addr #1
declare float @air.fast_pow.f32(float, float) local_unnamed_addr #1
declare float @air.fast_rint.f32(float) local_unnamed_addr #1
declare float @air.fast_round.f32(float) local_unnamed_addr #1
declare float @air.fast_rsqrt.f32(float) local_unnamed_addr #1
declare float @air.fast_sin.f32(float) local_unnamed_addr #1
declare float @air.fast_sinh.f32(float) local_unnamed_addr #1
declare float @air.fast_sqrt.f32(float) local_unnamed_addr #1
declare float @air.fast_tan.f32(float) local_unnamed_addr #1
declare float @air.fast_tanh.f32(float) local_unnamed_addr #1
declare float @air.sign.f32(float) local_unnamed_addr #1

define void @elementwise_f32(
%s) local_unnamed_addr #0 {
entry:
  %%idx32 = extractelement <3 x i32> %%gid, i64 0
  %%n_ptr = getelementptr inbounds %%struct.ElementwiseParams, %%struct.ElementwiseParams addrspace(2)* %%params, i64 0, i32 0
  %%n = load i32, i32 addrspace(2)* %%n_ptr, align 4
  %%in_bounds = icmp ult i32 %%idx32, %%n
  br i1 %%in_bounds, label %%body, label %%exit

body:
  %%idx = zext i32 %%idx32 to i64
%s
%s
  br label %%exit

exit:
  ret void
}

attributes #0 = { mustprogress nounwind "approx-func-fp-math"="true" "frame-pointer"="all" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { mustprogress nofree nosync nounwind readnone willreturn }

!air.kernel = !{!0}
!llvm.module.flags = !{%s}
!air.compile_options = !{%s}
!llvm.ident = !{!%d}
!air.version = !{!%d}
!air.language_version = !{!%d}
!air.source_file_name = !{!%d}

!0 = !{void (%s)* @elementwise_f32, !1, !2}
!1 = !{}
!2 = !{%s}
)",
                                         absl::StrJoin(args, ",\n"),
                                         absl::StrJoin(expression_body_, "\n"),
                                         store_result,
                                         absl::StrJoin(module_flag_refs, ", "),
                                         absl::StrJoin(compile_option_refs, ", "),
                                         ident_metadata, version_metadata,
                                         language_version_metadata,
                                         source_file_name_metadata,
                                         absl::StrJoin(signature_types, ", "),
                                         absl::StrJoin(metadata_args, ", "));

    for (int i = 0; i < parameter_numbers_.size(); ++i) {
      const PrimitiveType parameter_type = parameter_types_[i];
      absl::StrAppendFormat(
          &module,
          "!%d = !{i32 %d, !\"air.buffer\", !\"air.location_index\", i32 %d, "
          "i32 1, !\"air.read\", !\"air.address_space\", i32 1, "
          "!\"air.arg_type_size\", i32 %d, !\"air.arg_type_align_size\", i32 "
          "%d, !\"air.arg_type_name\", !\"%s\", !\"air.arg_name\", "
          "!\"arg%d\"}\n",
          3 + i, i, i, ElementTypeSize(parameter_type),
          ElementTypeSize(parameter_type), ElementAirTypeName(parameter_type),
          i);
    }
    absl::StrAppendFormat(
        &module,
        "!%d = !{i32 %d, !\"air.buffer\", !\"air.location_index\", i32 %d, "
        "i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, "
        "!\"air.arg_type_size\", i32 %d, !\"air.arg_type_align_size\", i32 %d, "
        "!\"air.arg_type_name\", !\"%s\", !\"air.arg_name\", !\"out\"}\n",
        output_metadata, static_cast<int>(parameter_numbers_.size()),
        static_cast<int>(parameter_numbers_.size()), ElementTypeSize(result_type),
        ElementTypeSize(result_type), ElementAirTypeName(result_type));
    absl::StrAppendFormat(
        &module,
        "!%d = !{i32 %d, !\"air.buffer\", !\"air.buffer_size\", i32 16, "
        "!\"air.location_index\", i32 %d, i32 1, !\"air.read\", "
        "!\"air.address_space\", i32 2, !\"air.struct_type_info\", !%d, "
        "!\"air.arg_type_size\", i32 16, !\"air.arg_type_align_size\", i32 4, "
        "!\"air.arg_type_name\", !\"ElementwiseParams\", !\"air.arg_name\", "
        "!\"params\"}\n",
        params_metadata, static_cast<int>(parameter_numbers_.size() + 1),
        static_cast<int>(parameter_numbers_.size() + 1), struct_metadata);
    absl::StrAppendFormat(
        &module,
        "!%d = !{i32 0, i32 4, i32 0, !\"uint\", !\"num_elements\", i32 4, "
        "i32 4, i32 0, !\"uint\", !\"reserved0\", i32 8, i32 4, i32 0, "
        "!\"uint\", !\"reserved1\", i32 12, i32 4, i32 0, !\"uint\", "
        "!\"reserved2\"}\n",
        struct_metadata);
    absl::StrAppendFormat(
        &module,
        "!%d = !{i32 %d, !\"air.thread_position_in_grid\", "
        "!\"air.arg_type_name\", !\"uint3\", !\"air.arg_name\", !\"gid\"}\n",
        gid_metadata, static_cast<int>(parameter_numbers_.size() + 2));
    absl::StrAppendFormat(
        &module,
        R"(
!%d = !{i32 1, !"wchar_size", i32 4}
!%d = !{i32 7, !"air.max_device_buffers", i32 31}
!%d = !{i32 7, !"air.max_constant_buffers", i32 31}
!%d = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!%d = !{i32 7, !"air.max_textures", i32 128}
!%d = !{i32 7, !"air.max_read_write_textures", i32 8}
!%d = !{i32 7, !"air.max_samplers", i32 16}
!%d = !{i32 7, !"frame-pointer", i32 2}
!%d = !{!"air.compile.denorms_disable"}
!%d = !{!"air.compile.fast_math_enable"}
!%d = !{!"air.compile.framebuffer_fetch_enable"}
!%d = !{!"xla direct AIR elementwise"}
!%d = !{i32 2, i32 7, i32 0}
!%d = !{!"Metal", i32 3, i32 2, i32 0}
!%d = !{!"xla/service/gpu/metal_elementwise_air"}
)",
        module_flags_metadata, module_flags_metadata + 1,
        module_flags_metadata + 2, module_flags_metadata + 3,
        module_flags_metadata + 4, module_flags_metadata + 5,
        module_flags_metadata + 6, module_flags_metadata + 7,
        compile_options_metadata, compile_options_metadata + 1,
        compile_options_metadata + 2, ident_metadata, version_metadata,
        language_version_metadata, source_file_name_metadata);
    return module;
  }

  Shape result_shape_;
  int64_t num_work_items_ = 0;
  bool raw_byte_result_ = false;
  bool raw_bf16_result_ = false;
  std::vector<CallParameterScope> call_parameter_scopes_;
  std::vector<ScalarParameterScope> scalar_parameter_scopes_;
  absl::flat_hash_map<int64_t, int> parameter_to_input_index_;
  std::vector<int64_t> parameter_numbers_;
  std::vector<PrimitiveType> parameter_types_;
  std::vector<std::string> expression_body_;
  std::string result_value_;
  int next_value_id_ = 0;
};

}  // namespace

absl::StatusOr<MetalMatmulConfig> MatchMetalMatmul(const HloModule& module) {
  const HloInstruction* root = module.entry_computation()->root_instruction();
  bool relu = false;
  TF_ASSIGN_OR_RETURN(const HloInstruction* dot, MatchDotRoot(root, &relu));

  const HloInstruction* lhs = dot->operand(0);
  const HloInstruction* rhs = dot->operand(1);
  const DotDimensionNumbers& dims = dot->dot_dimension_numbers();
  if (dims.lhs_batch_dimensions_size() != 0 ||
      dims.rhs_batch_dimensions_size() != 0 ||
      dims.lhs_contracting_dimensions_size() != 1 ||
      dims.rhs_contracting_dimensions_size() != 1) {
    return absl::UnimplementedError(
        "Metal direct AIR matmul supports only non-batched f32 dots with one "
        "contracting dimension.");
  }

  MetalMatmulConfig config;
  config.relu = relu;
  if (IsF32Rank1Array(lhs->shape()) && IsF32Rank1Array(rhs->shape()) &&
      IsScalarLikeF32(dot->shape())) {
    if (dims.lhs_contracting_dimensions(0) != 0 ||
        dims.rhs_contracting_dimensions(0) != 0) {
      return absl::UnimplementedError(
          "Metal direct AIR vector dot supports only lhs[0] x rhs[0].");
    }
    config.m = 1;
    config.n = 1;
    config.k = lhs->shape().dimensions(0);
    if (rhs->shape().dimensions(0) != config.k) {
      return absl::InvalidArgumentError(
          "Metal direct AIR vector dot dimensions are inconsistent.");
    }
    return config;
  }

  if (!IsF32Rank2Array(lhs->shape()) || !IsF32Rank2Array(rhs->shape()) ||
      !IsF32Rank2Array(dot->shape())) {
    return absl::UnimplementedError(
        "Metal direct AIR matmul supports only rank-1 f32 dot and rank-2 f32 "
        "matmul.");
  }
  if (dims.lhs_contracting_dimensions(0) != 1 ||
      dims.rhs_contracting_dimensions(0) != 0) {
    return absl::UnimplementedError(
        "Metal direct AIR matmul supports only non-batched row-major "
        "contracting dimensions lhs[1] x rhs[0].");
  }

  config.m = lhs->shape().dimensions(0);
  config.k = lhs->shape().dimensions(1);
  config.n = rhs->shape().dimensions(1);

  if (rhs->shape().dimensions(0) != config.k ||
      dot->shape().dimensions(0) != config.m ||
      dot->shape().dimensions(1) != config.n) {
    return absl::InvalidArgumentError(
        "Metal direct AIR matmul shape dimensions are inconsistent.");
  }

  config.use_simdgroup =
      config.m % 16 == 0 && config.n % 32 == 0 && config.k % 8 == 0;
  return config;
}

std::string BuildGenericMatmulAir(bool relu) {
  const char* kernel_name = relu ? "matmul_relu_scalar" : "matmul_scalar";
  std::string relu_epilogue;
  const char* result_value = "%acc";
  if (relu) {
    relu_epilogue = R"(  %relu_cmp = fcmp fast ogt float %acc, 0.000000e+00
  %relu_acc = select i1 %relu_cmp, float %acc, float 0.000000e+00
)";
    result_value = "%relu_acc";
  }

  return absl::StrCat(R"(
source_filename = "xla/service/gpu/metal_matmul_scalar_air"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024-n8:16:32"
target triple = "air64_v27-apple-macosx15.0.0"

%struct.MatmulParams = type { i32, i32, i32, i32 }

define void @)",
                      kernel_name, R"((
    float addrspace(1)* nocapture noundef readonly "air-buffer-no-alias" %a,
    float addrspace(1)* nocapture noundef readonly "air-buffer-no-alias" %b,
    float addrspace(1)* nocapture noundef writeonly "air-buffer-no-alias" %c,
    %struct.MatmulParams addrspace(2)* nocapture noundef readonly align 4 dereferenceable(16) "air-buffer-no-alias" %params,
    <3 x i32> noundef %gid) local_unnamed_addr #0 {
entry:
  %idx32 = extractelement <3 x i32> %gid, i64 0
  %m_ptr = getelementptr inbounds %struct.MatmulParams, %struct.MatmulParams addrspace(2)* %params, i64 0, i32 0
  %m = load i32, i32 addrspace(2)* %m_ptr, align 4
  %n_ptr = getelementptr inbounds %struct.MatmulParams, %struct.MatmulParams addrspace(2)* %params, i64 0, i32 1
  %n = load i32, i32 addrspace(2)* %n_ptr, align 4
  %k_ptr = getelementptr inbounds %struct.MatmulParams, %struct.MatmulParams addrspace(2)* %params, i64 0, i32 2
  %k = load i32, i32 addrspace(2)* %k_ptr, align 4
  %total = mul i32 %m, %n
  %in_bounds = icmp ult i32 %idx32, %total
  br i1 %in_bounds, label %body, label %exit

body:
  %row = udiv i32 %idx32, %n
  %col = urem i32 %idx32, %n
  br label %loop

loop:
  %kk = phi i32 [ 0, %body ], [ %next, %loop_body ]
  %acc = phi float [ 0.000000e+00, %body ], [ %acc_next, %loop_body ]
  %more_k = icmp ult i32 %kk, %k
  br i1 %more_k, label %loop_body, label %store

loop_body:
  %a_row_offset = mul i32 %row, %k
  %a_index32 = add i32 %a_row_offset, %kk
  %a_index = zext i32 %a_index32 to i64
  %a_ptr = getelementptr inbounds float, float addrspace(1)* %a, i64 %a_index
  %a_value = load float, float addrspace(1)* %a_ptr, align 4
  %b_row_offset = mul i32 %kk, %n
  %b_index32 = add i32 %b_row_offset, %col
  %b_index = zext i32 %b_index32 to i64
  %b_ptr = getelementptr inbounds float, float addrspace(1)* %b, i64 %b_index
  %b_value = load float, float addrspace(1)* %b_ptr, align 4
  %prod = fmul fast float %a_value, %b_value
  %acc_next = fadd fast float %acc, %prod
  %next = add i32 %kk, 1
  br label %loop

store:
)",
                      relu_epilogue, R"(  %idx = zext i32 %idx32 to i64
  %c_ptr = getelementptr inbounds float, float addrspace(1)* %c, i64 %idx
  store float )",
                      result_value, R"(, float addrspace(1)* %c_ptr, align 4
  br label %exit

exit:
  ret void
}

attributes #0 = { mustprogress nounwind "approx-func-fp-math"="true" "frame-pointer"="all" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!llvm.module.flags = !{!10, !11, !12, !13, !14, !15, !16, !17}
!air.compile_options = !{!18, !19, !20}
!llvm.ident = !{!21}
!air.version = !{!22}
!air.language_version = !{!23}
!air.source_file_name = !{!24}

!0 = !{void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, %struct.MatmulParams addrspace(2)*, <3 x i32>)* @)",
                      kernel_name, R"(, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6, !8}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"a"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"b"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"c"}
!6 = !{i32 3, !"air.buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !7, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"MatmulParams", !"air.arg_name", !"params"}
!7 = !{i32 0, i32 4, i32 0, !"uint", !"m", i32 4, i32 4, i32 0, !"uint", !"n", i32 8, i32 4, i32 0, !"uint", !"k", i32 12, i32 4, i32 0, !"uint", !"reserved"}
!8 = !{i32 4, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"gid"}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{i32 7, !"air.max_device_buffers", i32 31}
!12 = !{i32 7, !"air.max_constant_buffers", i32 31}
!13 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!14 = !{i32 7, !"air.max_textures", i32 128}
!15 = !{i32 7, !"air.max_read_write_textures", i32 8}
!16 = !{i32 7, !"air.max_samplers", i32 16}
!17 = !{i32 7, !"frame-pointer", i32 2}
!18 = !{!"air.compile.denorms_disable"}
!19 = !{!"air.compile.fast_math_enable"}
!20 = !{!"air.compile.framebuffer_fetch_enable"}
!21 = !{!"xla direct AIR generic matmul"}
!22 = !{i32 2, i32 7, i32 0}
!23 = !{!"Metal", i32 3, i32 2, i32 0}
!24 = !{!"xla/service/gpu/metal_matmul_scalar_air"}
)");
}

absl::StatusOr<std::vector<uint8_t>> CompileMetalMatmulAirToMetallib(
    const MetalMatmulConfig& config) {
  if (!config.use_simdgroup) {
    return CompileMetalAirToMetallib(BuildGenericMatmulAir(config.relu),
                                     "metal_matmul_scalar_air");
  }
  return CompileMetalAirToMetallib(get_matmul_air_direct(),
                                   "metal_matmul_air");
}

const char* ReductionValueIrType(PrimitiveType type) {
  switch (type) {
    case PRED:
      return "i1";
    case S32:
      return "i32";
    default:
      return "float";
  }
}

std::string ReductionScalarConstant(const HloInstruction* instr) {
  switch (instr->shape().element_type()) {
    case F32:
      return FloatLiteral(instr->literal().Get<float>({}));
    case PRED:
      return instr->literal().Get<bool>({}) ? "true" : "false";
    case S32:
      return absl::StrCat(instr->literal().Get<int32_t>({}));
    default:
      return "";
  }
}

class ReductionInputAirEmitter {
 public:
  explicit ReductionInputAirEmitter(MetalReductionConfig* config)
      : config_(config) {}

  absl::StatusOr<std::string> Emit(const HloInstruction* instr,
                                   absl::string_view index) {
    if (instr->opcode() == HloOpcode::kCall) {
      CallParameterScope scope;
      scope.computation = instr->to_apply();
      for (int64_t i = 0; i < instr->operand_count(); ++i) {
        scope.arguments[i] = instr->operand(i);
      }
      call_parameter_scopes_.push_back(std::move(scope));
      absl::Cleanup pop_scope = [this] { call_parameter_scopes_.pop_back(); };
      return Emit(instr->to_apply()->root_instruction(), index);
    }
    if (instr->opcode() == HloOpcode::kBitcast ||
        instr->opcode() == HloOpcode::kReshape) {
      if (ShapeUtil::ElementsIn(instr->shape()) !=
          ShapeUtil::ElementsIn(instr->operand(0)->shape())) {
        return absl::UnimplementedError(
            "Metal direct AIR reduction reshape must preserve element count.");
      }
      return Emit(instr->operand(0), index);
    }
    if (instr->opcode() == HloOpcode::kSlice) {
      if (instr->shape().dimensions().size() != 1 ||
          instr->operand(0)->shape().dimensions().size() != 1 ||
          instr->slice_strides().size() != 1 ||
          instr->shape().element_type() != config_->element_type) {
        return absl::UnimplementedError(
            "Metal direct AIR reduction slices must be rank-1 and preserve "
            "the reduced element type.");
      }
      std::string source_index = std::string(index);
      const int64_t stride = instr->slice_strides(0);
      const int64_t start = instr->slice_starts(0);
      if (stride != 1) {
        std::string scaled = NewName("reduce_slice_scaled");
        body_.push_back(absl::StrFormat("  %s = mul i64 %s, %d", scaled,
                                        source_index, stride));
        source_index = scaled;
      }
      if (start != 0) {
        std::string shifted = NewName("reduce_slice_index");
        body_.push_back(absl::StrFormat("  %s = add i64 %s, %d", shifted,
                                        source_index, start));
        source_index = shifted;
      }
      return Emit(instr->operand(0), source_index);
    }
    if (instr->opcode() == HloOpcode::kBroadcast &&
        ShapeUtil::IsEffectiveScalar(instr->operand(0)->shape())) {
      return Emit(instr->operand(0), "0");
    }
    if (instr->IsConstant() && ShapeUtil::IsEffectiveScalar(instr->shape())) {
      std::string constant = ReductionScalarConstant(instr);
      if (constant.empty()) {
        return absl::UnimplementedError(
            "Metal direct AIR reduction supports only f32/pred scalar "
            "constants.");
      }
      return constant;
    }
    if (instr->opcode() == HloOpcode::kParameter) {
      if (const HloInstruction* override = CallParameterOverride(instr)) {
        return Emit(override,
                    ShapeUtil::IsEffectiveScalar(override->shape()) ? "0"
                                                                    : index);
      }
      if (instr->shape().element_type() != config_->element_type ||
          !instr->shape().IsArray()) {
        return absl::UnimplementedError(
            "Metal direct AIR reduction parameters must be arrays matching the "
            "reduced element type.");
      }
      if (config_->parameter_number == -1) {
        config_->parameter_number = instr->parameter_number();
      } else if (config_->parameter_number != instr->parameter_number()) {
        return absl::UnimplementedError(
            "Metal direct AIR reduction currently supports one input "
            "parameter.");
      }
      return EmitLoad(index);
    }
    if (instr->shape().element_type() == F32) {
      return EmitF32Expression(instr, index);
    }
    if (instr->shape().element_type() == S32) {
      return EmitS32Expression(instr, index);
    }
    if (instr->shape().element_type() == PRED) {
      return EmitPredExpression(instr, index);
    }
    return absl::UnimplementedError(absl::StrFormat(
        "Metal direct AIR reduction input does not support HLO opcode %s.",
        HloOpcodeString(instr->opcode())));
  }

  const std::vector<std::string>& body() const { return body_; }

 private:
  struct CallParameterScope {
    const HloComputation* computation = nullptr;
    absl::flat_hash_map<int64_t, const HloInstruction*> arguments;
  };

  absl::StatusOr<std::string> EmitF32Expression(const HloInstruction* instr,
                                                absl::string_view index) {
    switch (instr->opcode()) {
      case HloOpcode::kAdd:
      case HloOpcode::kSubtract:
      case HloOpcode::kMultiply:
      case HloOpcode::kDivide:
      case HloOpcode::kMaximum:
      case HloOpcode::kMinimum:
        break;
      default:
        return absl::UnimplementedError(absl::StrFormat(
            "Metal direct AIR reduction f32 input does not support HLO opcode "
            "%s.",
            HloOpcodeString(instr->opcode())));
    }
    TF_ASSIGN_OR_RETURN(std::string lhs, Emit(instr->operand(0), index));
    TF_ASSIGN_OR_RETURN(std::string rhs, Emit(instr->operand(1), index));
    switch (instr->opcode()) {
      case HloOpcode::kAdd:
        return EmitOp("fadd fast float", lhs, rhs);
      case HloOpcode::kSubtract:
        return EmitOp("fsub fast float", lhs, rhs);
      case HloOpcode::kMultiply:
        return EmitOp("fmul fast float", lhs, rhs);
      case HloOpcode::kDivide:
        return EmitOp("fdiv fast float", lhs, rhs);
      case HloOpcode::kMaximum:
        return EmitCompareSelect("ogt", lhs, rhs);
      case HloOpcode::kMinimum:
        return EmitCompareSelect("olt", lhs, rhs);
      default:
        return absl::InternalError("Unexpected f32 reduction input opcode.");
    }
  }

  absl::StatusOr<std::string> EmitPredExpression(const HloInstruction* instr,
                                                 absl::string_view index) {
    if (instr->opcode() == HloOpcode::kCompare) {
      return EmitCompareExpression(instr, index);
    }
    switch (instr->opcode()) {
      case HloOpcode::kAnd:
      case HloOpcode::kOr:
      case HloOpcode::kXor:
        break;
      case HloOpcode::kNot: {
        TF_ASSIGN_OR_RETURN(std::string value, Emit(instr->operand(0), index));
        return EmitOp("xor i1", value, "true");
      }
      default:
        return absl::UnimplementedError(absl::StrFormat(
            "Metal direct AIR reduction pred input does not support HLO opcode "
            "%s.",
            HloOpcodeString(instr->opcode())));
    }
    TF_ASSIGN_OR_RETURN(std::string lhs, Emit(instr->operand(0), index));
    TF_ASSIGN_OR_RETURN(std::string rhs, Emit(instr->operand(1), index));
    switch (instr->opcode()) {
      case HloOpcode::kAnd:
        return EmitOp("and i1", lhs, rhs);
      case HloOpcode::kOr:
        return EmitOp("or i1", lhs, rhs);
      case HloOpcode::kXor:
        return EmitOp("xor i1", lhs, rhs);
      default:
        return absl::InternalError("Unexpected pred reduction input opcode.");
    }
  }

  absl::StatusOr<std::string> EmitS32Expression(const HloInstruction* instr,
                                                absl::string_view index) {
    if (instr->opcode() == HloOpcode::kSelect) {
      TF_ASSIGN_OR_RETURN(std::string pred, Emit(instr->operand(0), index));
      TF_ASSIGN_OR_RETURN(std::string on_true, Emit(instr->operand(1), index));
      TF_ASSIGN_OR_RETURN(std::string on_false, Emit(instr->operand(2), index));
      std::string value = NewName("reduce_select");
      body_.push_back(absl::StrFormat(
          "  %s = select i1 %s, i32 %s, i32 %s", value, pred, on_true,
          on_false));
      return value;
    }
    switch (instr->opcode()) {
      case HloOpcode::kAdd:
      case HloOpcode::kSubtract:
      case HloOpcode::kMultiply:
      case HloOpcode::kDivide:
      case HloOpcode::kRemainder:
      case HloOpcode::kMaximum:
      case HloOpcode::kMinimum:
        break;
      default:
        return absl::UnimplementedError(absl::StrFormat(
            "Metal direct AIR reduction s32 input does not support HLO opcode "
            "%s.",
            HloOpcodeString(instr->opcode())));
    }
    TF_ASSIGN_OR_RETURN(std::string lhs, Emit(instr->operand(0), index));
    TF_ASSIGN_OR_RETURN(std::string rhs, Emit(instr->operand(1), index));
    switch (instr->opcode()) {
      case HloOpcode::kAdd:
        return EmitOp("add i32", lhs, rhs);
      case HloOpcode::kSubtract:
        return EmitOp("sub i32", lhs, rhs);
      case HloOpcode::kMultiply:
        return EmitOp("mul i32", lhs, rhs);
      case HloOpcode::kDivide:
        return EmitOp("sdiv i32", lhs, rhs);
      case HloOpcode::kRemainder:
        return EmitOp("srem i32", lhs, rhs);
      case HloOpcode::kMaximum:
        return EmitIntCompareSelect("sgt", lhs, rhs);
      case HloOpcode::kMinimum:
        return EmitIntCompareSelect("slt", lhs, rhs);
      default:
        return absl::InternalError("Unexpected s32 reduction input opcode.");
    }
  }

  absl::StatusOr<std::string> EmitLoad(absl::string_view index) {
    const char* memory_type = ElementIrType(config_->element_type);
    std::string ptr = NewName("reduce_ptr");
    std::string loaded = NewName("reduce_loaded");
    body_.push_back(absl::StrFormat(
        "  %s = getelementptr inbounds %s, %s addrspace(1)* %%arg0, i64 %s",
        ptr, memory_type, memory_type, index));
    body_.push_back(absl::StrFormat(
        "  %s = load %s, %s addrspace(1)* %s, align %d", loaded, memory_type,
        memory_type, ptr, ElementTypeSize(config_->element_type)));
    if (config_->element_type == PRED) {
      std::string pred = NewName("reduce_pred");
      body_.push_back(
          absl::StrFormat("  %s = icmp ne i8 %s, 0", pred, loaded));
      return pred;
    }
    return loaded;
  }

  absl::StatusOr<std::string> EmitCompareExpression(
      const HloInstruction* instr, absl::string_view index) {
    TF_ASSIGN_OR_RETURN(std::string lhs, Emit(instr->operand(0), index));
    TF_ASSIGN_OR_RETURN(std::string rhs, Emit(instr->operand(1), index));
    const PrimitiveType operand_type = instr->operand(0)->shape().element_type();
    std::string cmp = NewName("reduce_cmp");
    if (operand_type == S32) {
      const char* predicate = nullptr;
      switch (instr->comparison_direction()) {
        case ComparisonDirection::kEq:
          predicate = "eq";
          break;
        case ComparisonDirection::kNe:
          predicate = "ne";
          break;
        case ComparisonDirection::kLt:
          predicate = "slt";
          break;
        case ComparisonDirection::kLe:
          predicate = "sle";
          break;
        case ComparisonDirection::kGt:
          predicate = "sgt";
          break;
        case ComparisonDirection::kGe:
          predicate = "sge";
          break;
      }
      body_.push_back(absl::StrFormat("  %s = icmp %s i32 %s, %s", cmp,
                                      predicate, lhs, rhs));
      return cmp;
    }
    if (operand_type == PRED) {
      const char* predicate = nullptr;
      switch (instr->comparison_direction()) {
        case ComparisonDirection::kEq:
          predicate = "eq";
          break;
        case ComparisonDirection::kNe:
          predicate = "ne";
          break;
        default:
          return absl::UnimplementedError(
              "Metal direct AIR reduction pred compare supports only EQ/NE.");
      }
      body_.push_back(absl::StrFormat("  %s = icmp %s i1 %s, %s", cmp,
                                      predicate, lhs, rhs));
      return cmp;
    }
    if (operand_type == F32) {
      const char* predicate = nullptr;
      switch (instr->comparison_direction()) {
        case ComparisonDirection::kEq:
          predicate = "oeq";
          break;
        case ComparisonDirection::kNe:
          predicate = "one";
          break;
        case ComparisonDirection::kLt:
          predicate = "olt";
          break;
        case ComparisonDirection::kLe:
          predicate = "ole";
          break;
        case ComparisonDirection::kGt:
          predicate = "ogt";
          break;
        case ComparisonDirection::kGe:
          predicate = "oge";
          break;
      }
      body_.push_back(absl::StrFormat("  %s = fcmp fast %s float %s, %s", cmp,
                                      predicate, lhs, rhs));
      return cmp;
    }
    return absl::UnimplementedError(
        "Metal direct AIR reduction compare supports only f32/s32/pred.");
  }

  std::string EmitOp(absl::string_view op, absl::string_view lhs,
                     absl::string_view rhs) {
    std::string name = NewName("reduce_op");
    body_.push_back(absl::StrFormat("  %s = %s %s, %s", name, op, lhs, rhs));
    return name;
  }

  std::string EmitCompareSelect(absl::string_view predicate,
                                absl::string_view lhs, absl::string_view rhs) {
    std::string cmp = NewName("reduce_cmp");
    std::string value = NewName("reduce_select");
    body_.push_back(absl::StrFormat("  %s = fcmp fast %s float %s, %s", cmp,
                                    predicate, lhs, rhs));
    body_.push_back(absl::StrFormat(
        "  %s = select i1 %s, float %s, float %s", value, cmp, lhs, rhs));
    return value;
  }

  std::string EmitIntCompareSelect(absl::string_view predicate,
                                   absl::string_view lhs,
                                   absl::string_view rhs) {
    std::string cmp = NewName("reduce_cmp");
    std::string value = NewName("reduce_select");
    body_.push_back(absl::StrFormat("  %s = icmp %s i32 %s, %s", cmp,
                                    predicate, lhs, rhs));
    body_.push_back(absl::StrFormat(
        "  %s = select i1 %s, i32 %s, i32 %s", value, cmp, lhs, rhs));
    return value;
  }

  std::string NewName(absl::string_view prefix) {
    return absl::StrFormat("%%%s%d", prefix, next_value_id_++);
  }

  const HloInstruction* CallParameterOverride(
      const HloInstruction* parameter) const {
    if (parameter->opcode() != HloOpcode::kParameter) {
      return nullptr;
    }
    for (auto it = call_parameter_scopes_.rbegin();
         it != call_parameter_scopes_.rend(); ++it) {
      if (parameter->parent() != it->computation) {
        continue;
      }
      auto found = it->arguments.find(parameter->parameter_number());
      if (found != it->arguments.end()) {
        return found->second;
      }
    }
    return nullptr;
  }

  MetalReductionConfig* config_;
  std::vector<CallParameterScope> call_parameter_scopes_;
  std::vector<std::string> body_;
  int next_value_id_ = 0;
};

absl::StatusOr<MetalReductionConfig> MatchMetalReduction(
    const HloModule& module) {
  const HloInstruction* root = module.entry_computation()->root_instruction();
  float output_scale = 1.0f;
  const HloInstruction* reduce = root;
  if (root->opcode() == HloOpcode::kDivide ||
      root->opcode() == HloOpcode::kMultiply) {
    const HloInstruction* lhs = root->operand(0);
    const HloInstruction* rhs = root->operand(1);
    if (lhs->opcode() == HloOpcode::kReduce && rhs->IsConstant() &&
        IsScalarLikeF32(rhs->shape())) {
      reduce = lhs;
      const float rhs_value = rhs->literal().Get<float>({});
      output_scale =
          root->opcode() == HloOpcode::kDivide ? 1.0f / rhs_value : rhs_value;
    } else if (root->opcode() == HloOpcode::kMultiply &&
               rhs->opcode() == HloOpcode::kReduce && lhs->IsConstant() &&
               IsScalarLikeF32(lhs->shape())) {
      reduce = rhs;
      output_scale = lhs->literal().Get<float>({});
    }
  }

  if (reduce->opcode() != HloOpcode::kReduce) {
    return absl::UnimplementedError(
        "Metal direct AIR reduction supports only reduce roots.");
  }
  if (reduce->operand_count() != 2 || reduce->dimensions().empty()) {
    return absl::UnimplementedError(
        "Metal direct AIR reduction requires at least one reduction "
        "dimension.");
  }

  const HloInstruction* input = reduce->operand(0);
  const HloInstruction* init = reduce->operand(1);
  const PrimitiveType element_type = input->shape().element_type();
  if ((element_type != F32 && element_type != PRED && element_type != S32) ||
      root->shape().element_type() != element_type ||
      !init->IsConstant() ||
      !ShapeUtil::IsEffectiveScalar(init->shape()) ||
      init->shape().element_type() != element_type) {
    return absl::UnimplementedError(
        "Metal direct AIR reduction currently supports f32/s32/pred inputs "
        "and matching scalar constant init values.");
  }

  const int64_t input_rank = input->shape().dimensions().size();
  auto reduces_all_dimensions = [&] {
    if (reduce->dimensions().size() != input_rank) {
      return false;
    }
    std::vector<bool> seen(input_rank, false);
    for (int64_t dim : reduce->dimensions()) {
      if (dim < 0 || dim >= input_rank || seen[dim]) {
        return false;
      }
      seen[dim] = true;
    }
    return true;
  };
  const int64_t reduction_dim =
      reduce->dimensions().size() == 1 ? reduce->dimensions()[0] : 0;
  bool rank2_to_rank1 = false;
  bool rank3_to_rank1 = false;
  int64_t reduce_count = 0;
  int64_t output_elements = 1;
  int64_t input_minor = 0;
  int64_t input_dim0 = 0;
  int64_t input_dim1 = 0;
  int64_t input_dim2 = 0;
  int64_t kept_dim = 0;
  if (ShapeUtil::IsEffectiveScalar(root->shape()) &&
      reduces_all_dimensions()) {
    reduce_count = ShapeUtil::ElementsIn(input->shape());
  } else if (reduce->dimensions().size() == 1 && input_rank == 2 &&
             root->shape().dimensions().size() == 1 &&
             reduction_dim >= 0 && reduction_dim < 2) {
    rank2_to_rank1 = true;
    input_minor = input->shape().dimensions(1);
    reduce_count = input->shape().dimensions(reduction_dim);
    const int64_t output_dim = reduction_dim == 0 ? input->shape().dimensions(1)
                                                  : input->shape().dimensions(0);
    if (root->shape().dimensions(0) != output_dim) {
      return absl::InvalidArgumentError(
          "Metal direct AIR rank-2 reduction output dimension is "
          "inconsistent.");
    }
    output_elements = output_dim;
  } else if (reduce->dimensions().size() == 2 && input_rank == 3 &&
             root->shape().dimensions().size() == 1) {
    std::vector<bool> reduced(input_rank, false);
    for (int64_t dim : reduce->dimensions()) {
      if (dim < 0 || dim >= input_rank || reduced[dim]) {
        return absl::InvalidArgumentError(
            "Metal direct AIR rank-3 reduction dimensions are invalid.");
      }
      reduced[dim] = true;
    }
    for (int64_t dim = 0; dim < input_rank; ++dim) {
      if (!reduced[dim]) {
        kept_dim = dim;
        break;
      }
    }
    rank3_to_rank1 = true;
    input_dim0 = input->shape().dimensions(0);
    input_dim1 = input->shape().dimensions(1);
    input_dim2 = input->shape().dimensions(2);
    reduce_count = 1;
    for (int64_t dim = 0; dim < input_rank; ++dim) {
      if (reduced[dim]) {
        reduce_count *= input->shape().dimensions(dim);
      }
    }
    output_elements = input->shape().dimensions(kept_dim);
    if (root->shape().dimensions(0) != output_elements) {
      return absl::InvalidArgumentError(
          "Metal direct AIR rank-3 reduction output dimension is "
          "inconsistent.");
    }
  } else {
    return absl::UnimplementedError(
        "Metal direct AIR reduction currently supports full-rank reductions "
        "to scalar, rank-2 to rank-1 reductions, and rank-3 to rank-1 "
        "reductions.");
  }

  const HloInstruction* reducer = reduce->to_apply()->root_instruction();
  ReductionKind kind;
  switch (reducer->opcode()) {
    case HloOpcode::kAdd:
      kind = ReductionKind::kAdd;
      break;
    case HloOpcode::kMultiply:
      kind = ReductionKind::kMultiply;
      break;
    case HloOpcode::kMaximum:
      kind = ReductionKind::kMaximum;
      break;
    case HloOpcode::kMinimum:
      kind = ReductionKind::kMinimum;
      break;
    case HloOpcode::kAnd:
      kind = ReductionKind::kAnd;
      break;
    case HloOpcode::kOr:
      kind = ReductionKind::kOr;
      break;
    default:
      return absl::UnimplementedError(absl::StrFormat(
          "Metal direct AIR reduction does not support reducer opcode %s.",
          HloOpcodeString(reducer->opcode())));
  }
  if (element_type == PRED &&
      kind != ReductionKind::kAnd && kind != ReductionKind::kOr) {
    return absl::UnimplementedError(
        "Metal direct AIR pred reductions currently support only and/or.");
  }
  if (element_type == F32 &&
      (kind == ReductionKind::kAnd || kind == ReductionKind::kOr)) {
    return absl::UnimplementedError(
        "Metal direct AIR f32 reductions do not support logical reducers.");
  }
  if (element_type == S32 &&
      (kind == ReductionKind::kAnd || kind == ReductionKind::kOr)) {
    return absl::UnimplementedError(
        "Metal direct AIR s32 reductions do not support logical reducers.");
  }

  MetalReductionConfig config;
  config.input = input;
  config.element_type = element_type;
  config.reduce_count = reduce_count;
  config.output_elements = output_elements;
  config.input_minor = input_minor;
  config.reduction_dim = reduction_dim;
  config.input_dim0 = input_dim0;
  config.input_dim1 = input_dim1;
  config.input_dim2 = input_dim2;
  config.kept_dim = kept_dim;
  config.init_value =
      element_type == F32 ? init->literal().Get<float>({}) : 0.0f;
  config.init_s32 =
      element_type == S32 ? init->literal().Get<int32_t>({}) : 0;
  config.init_pred =
      element_type == PRED ? init->literal().Get<bool>({}) : false;
  config.output_scale = output_scale;
  config.kind = kind;
  config.rank2_to_rank1 = rank2_to_rank1;
  config.rank3_to_rank1 = rank3_to_rank1;
  ReductionInputAirEmitter validator(&config);
  TF_ASSIGN_OR_RETURN(std::string unused, validator.Emit(input, "%i64"));
  if (config.parameter_number < 0) {
    return absl::UnimplementedError(
        "Metal direct AIR reduction requires a parameter-backed input.");
  }
  return config;
}

std::string ReductionUpdate(ReductionKind kind, PrimitiveType element_type) {
  switch (kind) {
    case ReductionKind::kAdd:
      if (element_type == S32) {
        return "  %new_acc = add i32 %acc, %value\n";
      }
      return "  %new_acc = fadd fast float %acc, %value\n";
    case ReductionKind::kMultiply:
      if (element_type == S32) {
        return "  %new_acc = mul i32 %acc, %value\n";
      }
      return "  %new_acc = fmul fast float %acc, %value\n";
    case ReductionKind::kMaximum:
      if (element_type == S32) {
        return R"(  %cmp = icmp sgt i32 %value, %acc
  %new_acc = select i1 %cmp, i32 %value, i32 %acc
)";
      }
      return R"(  %cmp = fcmp fast ogt float %value, %acc
  %new_acc = select i1 %cmp, float %value, float %acc
)";
    case ReductionKind::kMinimum:
      if (element_type == S32) {
        return R"(  %cmp = icmp slt i32 %value, %acc
  %new_acc = select i1 %cmp, i32 %value, i32 %acc
)";
      }
      return R"(  %cmp = fcmp fast olt float %value, %acc
  %new_acc = select i1 %cmp, float %value, float %acc
)";
    case ReductionKind::kAnd:
      return "  %new_acc = and i1 %acc, %value\n";
    case ReductionKind::kOr:
      return "  %new_acc = or i1 %acc, %value\n";
  }
  return "";
}

std::string ReductionInitValue(const MetalReductionConfig& config) {
  if (config.element_type == PRED) {
    return config.init_pred ? "true" : "false";
  }
  if (config.element_type == S32) {
    return absl::StrCat(config.init_s32);
  }
  return FloatLiteral(config.init_value);
}

std::string ReductionStore(const MetalReductionConfig& config,
                           absl::string_view out_ptr = "%out") {
  if (config.element_type == PRED) {
    return absl::StrFormat(R"(  %%out_i8 = zext i1 %%acc to i8
  store i8 %%out_i8, i8 addrspace(1)* %s, align 1)",
                           out_ptr);
  }
  if (config.element_type == S32) {
    return absl::StrFormat("  store i32 %%acc, i32 addrspace(1)* %s, align 4",
                           out_ptr);
  }
  return absl::StrFormat(
      R"(  %%scaled_acc = fmul fast float %%acc, %s
  store float %%scaled_acc, float addrspace(1)* %s, align 4)",
      FloatLiteral(config.output_scale), out_ptr);
}

std::string ReductionValueAssign(PrimitiveType element_type,
                                 absl::string_view value) {
  if (element_type == PRED) {
    return absl::StrFormat("  %%value = and i1 %s, true", value);
  }
  if (element_type == S32) {
    return absl::StrFormat("  %%value = add i32 %s, 0", value);
  }
  return absl::StrFormat(
      "  %%value = fadd fast float %s, 0x0000000000000000", value);
}

absl::StatusOr<std::string> BuildReductionAir(MetalReductionConfig config) {
  if (config.rank2_to_rank1 || config.rank3_to_rank1) {
    ReductionInputAirEmitter input_emitter(&config);
    TF_ASSIGN_OR_RETURN(std::string value,
                        input_emitter.Emit(config.input, "%source_i64"));
    const std::string input_body = absl::StrJoin(input_emitter.body(), "\n");
    std::string index_setup_body;
    std::string source_index_body;
    if (config.rank2_to_rank1) {
      index_setup_body = R"(  %minor_ptr = getelementptr inbounds %struct.ReductionParams, %struct.ReductionParams addrspace(2)* %params, i64 0, i32 2
  %minor = load i32, i32 addrspace(2)* %minor_ptr, align 4
  %dim_ptr = getelementptr inbounds %struct.ReductionParams, %struct.ReductionParams addrspace(2)* %params, i64 0, i32 3
  %dim = load i32, i32 addrspace(2)* %dim_ptr, align 4
  %reduce_dim0 = icmp eq i32 %dim, 0
)";
      source_index_body = R"(  %dim0_base = mul i32 %i, %minor
  %dim0_index = add i32 %dim0_base, %idx32
  %dim1_base = mul i32 %idx32, %minor
  %dim1_index = add i32 %dim1_base, %i
  %source32 = select i1 %reduce_dim0, i32 %dim0_index, i32 %dim1_index
  %source_i64 = zext i32 %source32 to i64
)";
    } else if (config.kept_dim == 0) {
      source_index_body = absl::StrFormat(
          R"(  %%kept_i64 = zext i32 %%idx32 to i64
  %%reduce_i64 = zext i32 %%i to i64
  %%rd0 = udiv i64 %%reduce_i64, %d
  %%rd1 = urem i64 %%reduce_i64, %d
  %%kept_offset = mul i64 %%kept_i64, %d
  %%rd0_offset = mul i64 %%rd0, %d
  %%partial_source = add i64 %%kept_offset, %%rd0_offset
  %%source_i64 = add i64 %%partial_source, %%rd1
)",
          config.input_dim2, config.input_dim2,
          config.input_dim1 * config.input_dim2, config.input_dim2);
    } else if (config.kept_dim == 1) {
      source_index_body = absl::StrFormat(
          R"(  %%kept_i64 = zext i32 %%idx32 to i64
  %%reduce_i64 = zext i32 %%i to i64
  %%rd0 = udiv i64 %%reduce_i64, %d
  %%rd1 = urem i64 %%reduce_i64, %d
  %%rd0_offset = mul i64 %%rd0, %d
  %%kept_offset = mul i64 %%kept_i64, %d
  %%partial_source = add i64 %%rd0_offset, %%kept_offset
  %%source_i64 = add i64 %%partial_source, %%rd1
)",
          config.input_dim2, config.input_dim2,
          config.input_dim1 * config.input_dim2, config.input_dim2);
    } else if (config.kept_dim == 2) {
      source_index_body = absl::StrFormat(
          R"(  %%kept_i64 = zext i32 %%idx32 to i64
  %%reduce_i64 = zext i32 %%i to i64
  %%rd0 = udiv i64 %%reduce_i64, %d
  %%rd1 = urem i64 %%reduce_i64, %d
  %%rd0_offset = mul i64 %%rd0, %d
  %%rd1_offset = mul i64 %%rd1, %d
  %%partial_source = add i64 %%rd0_offset, %%rd1_offset
  %%source_i64 = add i64 %%partial_source, %%kept_i64
)",
          config.input_dim1, config.input_dim1,
          config.input_dim1 * config.input_dim2, config.input_dim2);
    } else {
      return absl::InternalError(
          "Metal direct AIR rank-3 reduction kept dimension is invalid.");
    }
    const char* memory_type = ElementIrType(config.element_type);
    const char* value_type = ReductionValueIrType(config.element_type);
    const char* air_type = ElementAirTypeName(config.element_type);
    const char* kernel_name =
        config.element_type == PRED ? "reduce_pred" : "reduce_f32";
    return absl::StrFormat(R"(
source_filename = "xla/service/gpu/metal_reduction_air"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024-n8:16:32"
target triple = "air64_v27-apple-macosx15.0.0"

%%struct.ReductionParams = type { i32, i32, i32, i32 }

define void @%s(
    %s addrspace(1)* nocapture noundef readonly "air-buffer-no-alias" %%arg0,
    %s addrspace(1)* nocapture noundef writeonly "air-buffer-no-alias" %%out,
    %%struct.ReductionParams addrspace(2)* nocapture noundef readonly align 4 dereferenceable(16) "air-buffer-no-alias" %%params,
    <3 x i32> noundef %%gid) local_unnamed_addr #0 {
entry:
  %%idx32 = extractelement <3 x i32> %%gid, i64 0
  %%out_n_ptr = getelementptr inbounds %%struct.ReductionParams, %%struct.ReductionParams addrspace(2)* %%params, i64 0, i32 1
  %%out_n = load i32, i32 addrspace(2)* %%out_n_ptr, align 4
  %%in_bounds = icmp ult i32 %%idx32, %%out_n
  br i1 %%in_bounds, label %%body, label %%exit

body:
  %%reduce_n_ptr = getelementptr inbounds %%struct.ReductionParams, %%struct.ReductionParams addrspace(2)* %%params, i64 0, i32 0
  %%reduce_n = load i32, i32 addrspace(2)* %%reduce_n_ptr, align 4
%s
  br label %%loop

loop:
  %%i = phi i32 [ 0, %%body ], [ %%next, %%loop_body ]
  %%acc = phi %s [ %s, %%body ], [ %%new_acc, %%loop_body ]
  %%more = icmp ult i32 %%i, %%reduce_n
  br i1 %%more, label %%loop_body, label %%done

loop_body:
%s
%s
%s
%s  %%next = add i32 %%i, 1
  br label %%loop

done:
  %%out_i64 = zext i32 %%idx32 to i64
  %%out_ptr = getelementptr inbounds %s, %s addrspace(1)* %%out, i64 %%out_i64
%s
  br label %%exit

exit:
  ret void
}

attributes #0 = { mustprogress nounwind "approx-func-fp-math"="true" "frame-pointer"="all" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!llvm.module.flags = !{!10, !11, !12, !13, !14, !15, !16, !17}
!air.compile_options = !{!18, !19, !20}
!llvm.ident = !{!21}
!air.version = !{!22}
!air.language_version = !{!23}
!air.source_file_name = !{!24}

!0 = !{void (%s addrspace(1)*, %s addrspace(1)*, %%struct.ReductionParams addrspace(2)*, <3 x i32>)* @%s, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !7}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 %d, !"air.arg_type_align_size", i32 %d, !"air.arg_type_name", !"%s", !"air.arg_name", !"arg0"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 %d, !"air.arg_type_align_size", i32 %d, !"air.arg_type_name", !"%s", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !6, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"ReductionParams", !"air.arg_name", !"params"}
!6 = !{i32 0, i32 4, i32 0, !"uint", !"reduce_count", i32 4, i32 4, i32 0, !"uint", !"output_elements", i32 8, i32 4, i32 0, !"uint", !"input_minor", i32 12, i32 4, i32 0, !"uint", !"reduction_dim"}
!7 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"gid"}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{i32 7, !"air.max_device_buffers", i32 31}
!12 = !{i32 7, !"air.max_constant_buffers", i32 31}
!13 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!14 = !{i32 7, !"air.max_textures", i32 128}
!15 = !{i32 7, !"air.max_read_write_textures", i32 8}
!16 = !{i32 7, !"air.max_samplers", i32 16}
!17 = !{i32 7, !"frame-pointer", i32 2}
!18 = !{!"air.compile.denorms_disable"}
!19 = !{!"air.compile.fast_math_enable"}
!20 = !{!"air.compile.framebuffer_fetch_enable"}
!21 = !{!"xla direct AIR reduction"}
!22 = !{i32 2, i32 7, i32 0}
!23 = !{!"Metal", i32 3, i32 2, i32 0}
!24 = !{!"xla/service/gpu/metal_reduction_air"}
)",
                           kernel_name, memory_type, memory_type,
                           index_setup_body, value_type,
                           ReductionInitValue(config), source_index_body,
                           input_body,
                           ReductionValueAssign(config.element_type, value),
                           ReductionUpdate(config.kind, config.element_type),
                           memory_type, memory_type,
                           ReductionStore(config, "%out_ptr"), memory_type,
                           memory_type, kernel_name,
                           ElementTypeSize(config.element_type),
                           ElementTypeSize(config.element_type), air_type,
                           ElementTypeSize(config.element_type),
                           ElementTypeSize(config.element_type), air_type);
  }

  ReductionInputAirEmitter input_emitter(&config);
  TF_ASSIGN_OR_RETURN(std::string value,
                      input_emitter.Emit(config.input, "%i64"));
  const std::string input_body = absl::StrJoin(input_emitter.body(), "\n");
  const char* memory_type = ElementIrType(config.element_type);
  const char* value_type = ReductionValueIrType(config.element_type);
  const char* air_type = ElementAirTypeName(config.element_type);
  const char* kernel_name =
      config.element_type == PRED ? "reduce_pred" : "reduce_f32";
  return absl::StrFormat(R"(
source_filename = "xla/service/gpu/metal_reduction_air"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024-n8:16:32"
target triple = "air64_v27-apple-macosx15.0.0"

%%struct.ReductionParams = type { i32, i32, i32, i32 }

define void @%s(
    %s addrspace(1)* nocapture noundef readonly "air-buffer-no-alias" %%arg0,
    %s addrspace(1)* nocapture noundef writeonly "air-buffer-no-alias" %%out,
    %%struct.ReductionParams addrspace(2)* nocapture noundef readonly align 4 dereferenceable(16) "air-buffer-no-alias" %%params,
    <3 x i32> noundef %%gid) local_unnamed_addr #0 {
entry:
  %%idx32 = extractelement <3 x i32> %%gid, i64 0
  %%is_leader = icmp eq i32 %%idx32, 0
  br i1 %%is_leader, label %%body, label %%exit

body:
  %%n_ptr = getelementptr inbounds %%struct.ReductionParams, %%struct.ReductionParams addrspace(2)* %%params, i64 0, i32 0
  %%n = load i32, i32 addrspace(2)* %%n_ptr, align 4
  br label %%loop

loop:
  %%i = phi i32 [ 0, %%body ], [ %%next, %%loop_body ]
  %%acc = phi %s [ %s, %%body ], [ %%new_acc, %%loop_body ]
  %%in_bounds = icmp ult i32 %%i, %%n
  br i1 %%in_bounds, label %%loop_body, label %%done

loop_body:
  %%i64 = zext i32 %%i to i64
%s
%s
%s  %%next = add i32 %%i, 1
  br label %%loop

done:
%s
  br label %%exit

exit:
  ret void
}

attributes #0 = { mustprogress nounwind "approx-func-fp-math"="true" "frame-pointer"="all" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!llvm.module.flags = !{!10, !11, !12, !13, !14, !15, !16, !17}
!air.compile_options = !{!18, !19, !20}
!llvm.ident = !{!21}
!air.version = !{!22}
!air.language_version = !{!23}
!air.source_file_name = !{!24}

!0 = !{void (%s addrspace(1)*, %s addrspace(1)*, %%struct.ReductionParams addrspace(2)*, <3 x i32>)* @%s, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !7}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 %d, !"air.arg_type_align_size", i32 %d, !"air.arg_type_name", !"%s", !"air.arg_name", !"arg0"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 %d, !"air.arg_type_align_size", i32 %d, !"air.arg_type_name", !"%s", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !6, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"ReductionParams", !"air.arg_name", !"params"}
!6 = !{i32 0, i32 4, i32 0, !"uint", !"reduce_count", i32 4, i32 4, i32 0, !"uint", !"output_elements", i32 8, i32 4, i32 0, !"uint", !"input_minor", i32 12, i32 4, i32 0, !"uint", !"reduction_dim"}
!7 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"gid"}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{i32 7, !"air.max_device_buffers", i32 31}
!12 = !{i32 7, !"air.max_constant_buffers", i32 31}
!13 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!14 = !{i32 7, !"air.max_textures", i32 128}
!15 = !{i32 7, !"air.max_read_write_textures", i32 8}
!16 = !{i32 7, !"air.max_samplers", i32 16}
!17 = !{i32 7, !"frame-pointer", i32 2}
!18 = !{!"air.compile.denorms_disable"}
!19 = !{!"air.compile.fast_math_enable"}
!20 = !{!"air.compile.framebuffer_fetch_enable"}
!21 = !{!"xla direct AIR reduction"}
!22 = !{i32 2, i32 7, i32 0}
!23 = !{!"Metal", i32 3, i32 2, i32 0}
!24 = !{!"xla/service/gpu/metal_reduction_air"}
)",
                         kernel_name, memory_type, memory_type, value_type,
                         ReductionInitValue(config), input_body,
                         ReductionValueAssign(config.element_type, value),
                         ReductionUpdate(config.kind, config.element_type),
                         ReductionStore(config), memory_type, memory_type,
                         kernel_name, ElementTypeSize(config.element_type),
                         ElementTypeSize(config.element_type), air_type,
                         ElementTypeSize(config.element_type),
                         ElementTypeSize(config.element_type), air_type);
}

class MetalReductionExecutable final : public Executable {
 public:
  MetalReductionExecutable(std::shared_ptr<HloModule> module,
                           MetalReductionConfig config,
                           std::vector<uint8_t> metallib)
      : Executable(std::move(module)),
        config_(config),
        result_shape_(this->module().output_shape()),
        metallib_(std::move(metallib)) {}

  Shape result_shape() const override { return result_shape_; }

  absl::StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments) override {
    se::Stream* stream = run_options->stream();
    if (stream == nullptr) {
      return absl::InvalidArgumentError("Metal reduction requires a stream.");
    }
    se::StreamExecutor* executor = stream->parent();
    se::DeviceAddressAllocator* allocator = run_options->allocator();
    if (allocator == nullptr) {
      return absl::InvalidArgumentError(
          "Metal reduction requires an allocator.");
    }
    if (config_.parameter_number >= arguments.size()) {
      return absl::InvalidArgumentError(
          "Metal reduction parameter number is out of bounds.");
    }

    const int device_ordinal = run_options->device_ordinal() != -1
                                   ? run_options->device_ordinal()
                                   : executor->device_ordinal();
    const int64_t output_size = ShapeUtil::ByteSizeOf(result_shape_, 8);
    TF_ASSIGN_OR_RETURN(se::ScopedDeviceAddress<uint8_t> output_buffer,
                        allocator->Allocate(device_ordinal, output_size));
    se::DeviceAddressBase output = *output_buffer;

    TF_ASSIGN_OR_RETURN(se::ScopedDeviceAddress<uint8_t> params_buffer,
                        allocator->Allocate(device_ordinal,
                                            sizeof(ReductionParams)));
    se::DeviceAddressBase params_address = *params_buffer;
    ReductionParams params{static_cast<uint32_t>(config_.reduce_count),
                           static_cast<uint32_t>(config_.output_elements),
                           static_cast<uint32_t>(config_.input_minor),
                           static_cast<uint32_t>(config_.reduction_dim)};
    TF_RETURN_IF_ERROR(
        stream->Memcpy(&params_address, &params, sizeof(ReductionParams)));

    se::DeviceAddressBase input =
        arguments[config_.parameter_number].Buffer({}).AsDeviceAddress();
    if (input.is_null()) {
      return absl::InvalidArgumentError("Metal reduction input buffer is null.");
    }

    const char* kernel_name =
        config_.element_type == PRED ? "reduce_pred" : "reduce_f32";
    auto spec = se::KernelLoaderSpec::CreateOwningMetalLibraryInMemorySpec(
        std::vector<uint8_t>(metallib_.begin(), metallib_.end()), kernel_name,
        /*arity=*/3);
    TF_ASSIGN_OR_RETURN(std::unique_ptr<se::Kernel> kernel,
                        executor->LoadKernel(spec));

    se::KernelArgsPackedArray kernel_args(/*num_args=*/3);
    kernel_args.add_argument(input);
    kernel_args.add_argument(output);
    kernel_args.add_argument(params_address);

    const uint64_t output_elements =
        static_cast<uint64_t>(config_.output_elements);
    const bool vector_output = config_.rank2_to_rank1 || config_.rank3_to_rank1;
    const se::ThreadDim threads(vector_output ? 256 : 1, 1, 1);
    const se::BlockDim blocks(vector_output
                                  ? (output_elements + 255) / 256
                                  : 1,
                              1, 1);
    TF_RETURN_IF_ERROR(kernel->Launch(threads, blocks, stream, kernel_args));
    TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());

    ExecutionOutput result(result_shape_, allocator, device_ordinal,
                           executor->device_ordinal());
    static_cast<ShapedBuffer*>(result.MutableResult())
        ->set_buffer(output_buffer.Release(), {});
    result.Commit();
    return std::move(result);
  }

 private:
  MetalReductionConfig config_;
  Shape result_shape_;
  std::vector<uint8_t> metallib_;
};

absl::StatusOr<std::unique_ptr<Executable>> BuildMetalReductionExecutable(
    std::shared_ptr<HloModule> module) {
  TF_ASSIGN_OR_RETURN(MetalReductionConfig config,
                      MatchMetalReduction(*module));
  TF_ASSIGN_OR_RETURN(std::string air, BuildReductionAir(config));
  TF_ASSIGN_OR_RETURN(std::vector<uint8_t> metallib,
                      CompileMetalAirToMetallib(air, "metal_reduction_air"));
  return std::make_unique<MetalReductionExecutable>(
      std::move(module), config, std::move(metallib));
}

absl::StatusOr<MetalConvertConfig> MatchMetalConvert(const HloModule& module) {
  const HloInstruction* root = module.entry_computation()->root_instruction();
  if (root->opcode() != HloOpcode::kConvert) {
    return absl::UnimplementedError(
        "Metal direct AIR convert supports only convert roots.");
  }
  const HloInstruction* input = root->operand(0);
  if (input->opcode() != HloOpcode::kParameter ||
      root->shape().dimensions() != input->shape().dimensions()) {
    return absl::UnimplementedError(
        "Metal direct AIR convert currently supports only parameter operands "
        "with unchanged dimensions.");
  }

  MetalConvertConfig config;
  config.parameter_number = input->parameter_number();
  config.num_elements = ShapeUtil::ElementsIn(root->shape());
  if (IsF32Array(input->shape()) && IsS32Array(root->shape())) {
    config.kind = ConvertKind::kF32ToS32;
    return config;
  }
  if (IsS32Array(input->shape()) && IsF32Array(root->shape())) {
    config.kind = ConvertKind::kS32ToF32;
    return config;
  }
  return absl::UnimplementedError(
      "Metal direct AIR convert currently supports only f32<->s32 arrays.");
}

std::string BuildConvertAir(const MetalConvertConfig& config) {
  const bool f32_to_s32 = config.kind == ConvertKind::kF32ToS32;
  const char* input_ir_type = f32_to_s32 ? "float" : "i32";
  const char* output_ir_type = f32_to_s32 ? "i32" : "float";
  const char* input_air_type = f32_to_s32 ? "float" : "int";
  const char* output_air_type = f32_to_s32 ? "int" : "float";
  const char* convert_op = f32_to_s32 ? "fptosi float %value to i32"
                                      : "sitofp i32 %value to float";
  return absl::StrFormat(R"(
source_filename = "xla/service/gpu/metal_convert_air"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024-n8:16:32"
target triple = "air64_v27-apple-macosx15.0.0"

%%struct.ConvertParams = type { i32, i32, i32, i32 }

define void @convert_elementwise(
    %s addrspace(1)* nocapture noundef readonly "air-buffer-no-alias" %%arg0,
    %s addrspace(1)* nocapture noundef writeonly "air-buffer-no-alias" %%out,
    %%struct.ConvertParams addrspace(2)* nocapture noundef readonly align 4 dereferenceable(16) "air-buffer-no-alias" %%params,
    <3 x i32> noundef %%gid) local_unnamed_addr #0 {
entry:
  %%idx32 = extractelement <3 x i32> %%gid, i64 0
  %%n_ptr = getelementptr inbounds %%struct.ConvertParams, %%struct.ConvertParams addrspace(2)* %%params, i64 0, i32 0
  %%n = load i32, i32 addrspace(2)* %%n_ptr, align 4
  %%in_bounds = icmp ult i32 %%idx32, %%n
  br i1 %%in_bounds, label %%body, label %%exit

body:
  %%idx = zext i32 %%idx32 to i64
  %%ptr = getelementptr inbounds %s, %s addrspace(1)* %%arg0, i64 %%idx
  %%value = load %s, %s addrspace(1)* %%ptr, align 4
  %%converted = %s
  %%out_ptr = getelementptr inbounds %s, %s addrspace(1)* %%out, i64 %%idx
  store %s %%converted, %s addrspace(1)* %%out_ptr, align 4
  br label %%exit

exit:
  ret void
}

attributes #0 = { mustprogress nounwind "approx-func-fp-math"="true" "frame-pointer"="all" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }

!air.kernel = !{!0}
!llvm.module.flags = !{!10, !11, !12, !13, !14, !15, !16, !17}
!air.compile_options = !{!18, !19, !20}
!llvm.ident = !{!21}
!air.version = !{!22}
!air.language_version = !{!23}
!air.source_file_name = !{!24}

!0 = !{void (%s addrspace(1)*, %s addrspace(1)*, %%struct.ConvertParams addrspace(2)*, <3 x i32>)* @convert_elementwise, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !7}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"%s", !"air.arg_name", !"arg0"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"%s", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !6, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"ConvertParams", !"air.arg_name", !"params"}
!6 = !{i32 0, i32 4, i32 0, !"uint", !"num_elements", i32 4, i32 4, i32 0, !"uint", !"reserved0", i32 8, i32 4, i32 0, !"uint", !"reserved1", i32 12, i32 4, i32 0, !"uint", !"reserved2"}
!7 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"gid"}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{i32 7, !"air.max_device_buffers", i32 31}
!12 = !{i32 7, !"air.max_constant_buffers", i32 31}
!13 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!14 = !{i32 7, !"air.max_textures", i32 128}
!15 = !{i32 7, !"air.max_read_write_textures", i32 8}
!16 = !{i32 7, !"air.max_samplers", i32 16}
!17 = !{i32 7, !"frame-pointer", i32 2}
!18 = !{!"air.compile.denorms_disable"}
!19 = !{!"air.compile.fast_math_enable"}
!20 = !{!"air.compile.framebuffer_fetch_enable"}
!21 = !{!"xla direct AIR convert"}
!22 = !{i32 2, i32 7, i32 0}
!23 = !{!"Metal", i32 3, i32 2, i32 0}
!24 = !{!"xla/service/gpu/metal_convert_air"}
)",
                         input_ir_type, output_ir_type, input_ir_type,
                         input_ir_type, input_ir_type, input_ir_type,
                         convert_op, output_ir_type, output_ir_type,
                         output_ir_type, output_ir_type, input_ir_type,
                         output_ir_type, input_air_type, output_air_type);
}

class MetalConvertExecutable final : public Executable {
 public:
  MetalConvertExecutable(std::shared_ptr<HloModule> module,
                         MetalConvertConfig config,
                         std::vector<uint8_t> metallib)
      : Executable(std::move(module)),
        config_(config),
        result_shape_(this->module().output_shape()),
        metallib_(std::move(metallib)) {}

  Shape result_shape() const override { return result_shape_; }

  absl::StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments) override {
    se::Stream* stream = run_options->stream();
    if (stream == nullptr) {
      return absl::InvalidArgumentError("Metal convert requires a stream.");
    }
    se::StreamExecutor* executor = stream->parent();
    se::DeviceAddressAllocator* allocator = run_options->allocator();
    if (allocator == nullptr) {
      return absl::InvalidArgumentError("Metal convert requires an allocator.");
    }
    if (config_.parameter_number >= arguments.size()) {
      return absl::InvalidArgumentError(
          "Metal convert parameter number is out of bounds.");
    }

    const int device_ordinal = run_options->device_ordinal() != -1
                                   ? run_options->device_ordinal()
                                   : executor->device_ordinal();
    const int64_t output_size = ShapeUtil::ByteSizeOf(result_shape_, 8);
    TF_ASSIGN_OR_RETURN(se::ScopedDeviceAddress<uint8_t> output_buffer,
                        allocator->Allocate(device_ordinal, output_size));
    se::DeviceAddressBase output = *output_buffer;

    TF_ASSIGN_OR_RETURN(se::ScopedDeviceAddress<uint8_t> params_buffer,
                        allocator->Allocate(device_ordinal,
                                            sizeof(ConvertParams)));
    se::DeviceAddressBase params_address = *params_buffer;
    ConvertParams params{static_cast<uint32_t>(config_.num_elements), 0, 0, 0};
    TF_RETURN_IF_ERROR(
        stream->Memcpy(&params_address, &params, sizeof(ConvertParams)));

    se::DeviceAddressBase input =
        arguments[config_.parameter_number].Buffer({}).AsDeviceAddress();
    if (input.is_null()) {
      return absl::InvalidArgumentError("Metal convert input buffer is null.");
    }

    auto spec = se::KernelLoaderSpec::CreateOwningMetalLibraryInMemorySpec(
        std::vector<uint8_t>(metallib_.begin(), metallib_.end()),
        "convert_elementwise", /*arity=*/3);
    TF_ASSIGN_OR_RETURN(std::unique_ptr<se::Kernel> kernel,
                        executor->LoadKernel(spec));

    se::KernelArgsPackedArray kernel_args(/*num_args=*/3);
    kernel_args.add_argument(input);
    kernel_args.add_argument(output);
    kernel_args.add_argument(params_address);

    se::ThreadDim threads(/*x=*/256, /*y=*/1, /*z=*/1);
    se::BlockDim blocks(/*x=*/static_cast<uint64_t>((config_.num_elements + 255) /
                                                    256),
                        /*y=*/1, /*z=*/1);
    TF_RETURN_IF_ERROR(kernel->Launch(threads, blocks, stream, kernel_args));
    TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());

    ExecutionOutput result(result_shape_, allocator, device_ordinal,
                           executor->device_ordinal());
    static_cast<ShapedBuffer*>(result.MutableResult())
        ->set_buffer(output_buffer.Release(), {});
    result.Commit();
    return std::move(result);
  }

 private:
  MetalConvertConfig config_;
  Shape result_shape_;
  std::vector<uint8_t> metallib_;
};

absl::StatusOr<std::unique_ptr<Executable>> BuildMetalConvertExecutable(
    std::shared_ptr<HloModule> module) {
  TF_ASSIGN_OR_RETURN(MetalConvertConfig config, MatchMetalConvert(*module));
  TF_ASSIGN_OR_RETURN(std::vector<uint8_t> metallib,
                      CompileMetalAirToMetallib(BuildConvertAir(config),
                                                "metal_convert_air"));
  return std::make_unique<MetalConvertExecutable>(
      std::move(module), config, std::move(metallib));
}

class MetalElementwiseExecutable final : public Executable {
 public:
  MetalElementwiseExecutable(std::shared_ptr<HloModule> module,
                             std::vector<int64_t> parameter_numbers,
                             int64_t num_elements,
                             std::vector<uint8_t> metallib)
      : Executable(std::move(module)),
        parameter_numbers_(std::move(parameter_numbers)),
        result_shape_(this->module().output_shape()),
        num_elements_(num_elements),
        metallib_(std::move(metallib)) {}

  Shape result_shape() const override { return result_shape_; }

  absl::StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments) override {
    se::Stream* stream = run_options->stream();
    if (stream == nullptr) {
      return absl::InvalidArgumentError("Metal elementwise requires a stream.");
    }
    se::StreamExecutor* executor = stream->parent();
    se::DeviceAddressAllocator* allocator = run_options->allocator();
    if (allocator == nullptr) {
      return absl::InvalidArgumentError(
          "Metal elementwise requires an allocator.");
    }

    const int device_ordinal = run_options->device_ordinal() != -1
                                   ? run_options->device_ordinal()
                                   : executor->device_ordinal();
    const int64_t output_size = ShapeUtil::ByteSizeOf(result_shape_, 8);
    TF_ASSIGN_OR_RETURN(se::ScopedDeviceAddress<uint8_t> output_buffer,
                        allocator->Allocate(device_ordinal, output_size));
    se::DeviceAddressBase output = *output_buffer;

    if (num_elements_ == 0) {
      ExecutionOutput result(result_shape_, allocator, device_ordinal,
                             executor->device_ordinal());
      static_cast<ShapedBuffer*>(result.MutableResult())
          ->set_buffer(output_buffer.Release(), {});
      result.Commit();
      return std::move(result);
    }

    TF_ASSIGN_OR_RETURN(se::ScopedDeviceAddress<uint8_t> params_buffer,
                        allocator->Allocate(device_ordinal,
                                            sizeof(ElementwiseParams)));
    se::DeviceAddressBase params_address = *params_buffer;
    ElementwiseParams params{static_cast<uint32_t>(num_elements_), 0, 0, 0};
    TF_RETURN_IF_ERROR(
        stream->Memcpy(&params_address, &params, sizeof(ElementwiseParams)));

    auto spec = se::KernelLoaderSpec::CreateOwningMetalLibraryInMemorySpec(
        std::vector<uint8_t>(metallib_.begin(), metallib_.end()),
        "elementwise_f32", parameter_numbers_.size() + 2);
    TF_ASSIGN_OR_RETURN(std::unique_ptr<se::Kernel> kernel,
                        executor->LoadKernel(spec));

    se::KernelArgsPackedArray kernel_args(parameter_numbers_.size() + 2);
    for (int64_t parameter_number : parameter_numbers_) {
      if (parameter_number >= arguments.size()) {
        return absl::InvalidArgumentError(
            "Metal elementwise parameter number is out of bounds.");
      }
      se::DeviceAddressBase arg =
          arguments[parameter_number].Buffer({}).AsDeviceAddress();
      if (arg.is_null()) {
        return absl::InvalidArgumentError(
            "Metal elementwise input buffer is null.");
      }
      kernel_args.add_argument(arg);
    }
    kernel_args.add_argument(output);
    kernel_args.add_argument(params_address);

    se::ThreadDim threads(/*x=*/256, /*y=*/1, /*z=*/1);
    se::BlockDim blocks(/*x=*/static_cast<uint64_t>((num_elements_ + 255) / 256),
                        /*y=*/1, /*z=*/1);
    TF_RETURN_IF_ERROR(kernel->Launch(threads, blocks, stream, kernel_args));
    TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());

    ExecutionOutput result(result_shape_, allocator, device_ordinal,
                           executor->device_ordinal());
    static_cast<ShapedBuffer*>(result.MutableResult())
        ->set_buffer(output_buffer.Release(), {});
    result.Commit();
    return std::move(result);
  }

 private:
  std::vector<int64_t> parameter_numbers_;
  Shape result_shape_;
  int64_t num_elements_;
  std::vector<uint8_t> metallib_;
};

class MetalTupleElementwiseExecutable final : public Executable {
 public:
  MetalTupleElementwiseExecutable(
      std::shared_ptr<HloModule> module,
      std::vector<std::vector<int64_t>> parameter_numbers,
      std::vector<int64_t> num_elements,
      std::vector<std::vector<uint8_t>> metallibs)
      : Executable(std::move(module)),
        parameter_numbers_(std::move(parameter_numbers)),
        result_shape_(this->module().output_shape()),
        num_elements_(std::move(num_elements)),
        metallibs_(std::move(metallibs)) {}

  Shape result_shape() const override { return result_shape_; }

  absl::StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments) override {
    se::Stream* stream = run_options->stream();
    if (stream == nullptr) {
      return absl::InvalidArgumentError(
          "Metal tuple elementwise requires a stream.");
    }
    se::StreamExecutor* executor = stream->parent();
    se::DeviceAddressAllocator* allocator = run_options->allocator();
    if (allocator == nullptr) {
      return absl::InvalidArgumentError(
          "Metal tuple elementwise requires an allocator.");
    }
    TF_ASSIGN_OR_RETURN(TransferManager * transfer_manager,
                        TransferManager::GetForPlatform(executor->GetPlatform()));

    const int device_ordinal = run_options->device_ordinal() != -1
                                   ? run_options->device_ordinal()
                                   : executor->device_ordinal();
    const int64_t tuple_element_count =
        ShapeUtil::TupleElementCount(result_shape_);
    TF_ASSIGN_OR_RETURN(
        se::ScopedDeviceAddress<uint8_t> tuple_buffer,
        allocator->Allocate(device_ordinal,
                            transfer_manager->GetByteSizeRequirement(
                                result_shape_)));

    std::vector<se::ScopedDeviceAddress<uint8_t>> element_buffers;
    std::vector<se::ScopedDeviceAddress<uint8_t>> params_buffers;
    element_buffers.reserve(tuple_element_count);
    params_buffers.reserve(tuple_element_count);

    for (int64_t i = 0; i < tuple_element_count; ++i) {
      const Shape& element_shape = result_shape_.tuple_shapes(i);
      TF_ASSIGN_OR_RETURN(
          se::ScopedDeviceAddress<uint8_t> element_buffer,
          allocator->Allocate(device_ordinal,
                              transfer_manager->GetByteSizeRequirement(
                                  element_shape)));
      se::DeviceAddressBase output = *element_buffer;

      if (num_elements_[i] != 0) {
        TF_ASSIGN_OR_RETURN(se::ScopedDeviceAddress<uint8_t> params_buffer,
                            allocator->Allocate(device_ordinal,
                                                sizeof(ElementwiseParams)));
        se::DeviceAddressBase params_address = *params_buffer;
        ElementwiseParams params{static_cast<uint32_t>(num_elements_[i]), 0, 0,
                                 0};
        TF_RETURN_IF_ERROR(stream->Memcpy(&params_address, &params,
                                          sizeof(ElementwiseParams)));
        TF_RETURN_IF_ERROR(LaunchElementKernel(executor, stream, i, arguments,
                                               output, params_address));
        params_buffers.push_back(std::move(params_buffer));
      }
      element_buffers.push_back(std::move(element_buffer));
    }

    ExecutionOutput result(result_shape_, allocator, device_ordinal,
                           executor->device_ordinal());
    ScopedShapedBuffer* shaped_result = result.MutableResult();
    ShapedBuffer* shaped_buffer = static_cast<ShapedBuffer*>(shaped_result);
    shaped_buffer->set_buffer(tuple_buffer.Release(), {});
    for (int64_t i = 0; i < tuple_element_count; ++i) {
      shaped_buffer->set_buffer(element_buffers[i].Release(), {i});
    }
    TF_RETURN_IF_ERROR(
        transfer_manager->WriteRootTupleIndexTable(stream, *shaped_result));
    TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());

    result.Commit();
    return std::move(result);
  }

 private:
  absl::Status LaunchElementKernel(
      se::StreamExecutor* executor, se::Stream* stream, int64_t tuple_index,
      const std::vector<ExecutionInput>& arguments, se::DeviceAddressBase output,
      se::DeviceAddressBase params_address) const {
    auto spec = se::KernelLoaderSpec::CreateOwningMetalLibraryInMemorySpec(
        std::vector<uint8_t>(metallibs_[tuple_index].begin(),
                             metallibs_[tuple_index].end()),
        "elementwise_f32", parameter_numbers_[tuple_index].size() + 2);
    TF_ASSIGN_OR_RETURN(std::unique_ptr<se::Kernel> kernel,
                        executor->LoadKernel(spec));

    se::KernelArgsPackedArray kernel_args(
        parameter_numbers_[tuple_index].size() + 2);
    for (int64_t parameter_number : parameter_numbers_[tuple_index]) {
      if (parameter_number < 0 || parameter_number >= arguments.size()) {
        return absl::InvalidArgumentError(
            "Metal tuple elementwise parameter number is out of bounds.");
      }
      se::DeviceAddressBase arg =
          arguments[parameter_number].Buffer({}).AsDeviceAddress();
      if (arg.is_null()) {
        return absl::InvalidArgumentError(
            "Metal tuple elementwise input buffer is null.");
      }
      kernel_args.add_argument(arg);
    }
    kernel_args.add_argument(output);
    kernel_args.add_argument(params_address);

    se::ThreadDim threads(/*x=*/256, /*y=*/1, /*z=*/1);
    se::BlockDim blocks(
        /*x=*/static_cast<uint64_t>((num_elements_[tuple_index] + 255) / 256),
        /*y=*/1, /*z=*/1);
    return kernel->Launch(threads, blocks, stream, kernel_args);
  }

  std::vector<std::vector<int64_t>> parameter_numbers_;
  Shape result_shape_;
  std::vector<int64_t> num_elements_;
  std::vector<std::vector<uint8_t>> metallibs_;
};

absl::StatusOr<std::unique_ptr<Executable>>
BuildMetalTupleElementwiseExecutable(std::shared_ptr<HloModule> module) {
  HloInstruction* root = module->entry_computation()->root_instruction();
  if (root->opcode() != HloOpcode::kTuple || !root->shape().IsTuple() ||
      root->operand_count() != ShapeUtil::TupleElementCount(root->shape())) {
    return absl::UnimplementedError(
        "Metal direct AIR tuple elementwise executable requires a tuple root.");
  }

  std::vector<std::vector<int64_t>> parameter_numbers;
  std::vector<int64_t> num_elements;
  std::vector<std::vector<uint8_t>> metallibs;
  const int64_t tuple_element_count =
      ShapeUtil::TupleElementCount(root->shape());
  parameter_numbers.reserve(tuple_element_count);
  num_elements.reserve(tuple_element_count);
  metallibs.reserve(tuple_element_count);
  for (int64_t i = 0; i < tuple_element_count; ++i) {
    const Shape& element_shape = root->shape().tuple_shapes(i);
    if (!element_shape.IsArray() ||
        !ShapeUtil::Compatible(element_shape, root->operand(i)->shape())) {
      return absl::UnimplementedError(
          "Metal direct AIR tuple elementwise executable supports only tuple "
          "operands matching array tuple element shapes.");
    }
    ElementwiseAirEmitter emitter(element_shape);
    TF_ASSIGN_OR_RETURN(std::string air, emitter.Emit(root->operand(i)));
    TF_ASSIGN_OR_RETURN(
        std::vector<uint8_t> metallib,
        CompileMetalAirToMetallib(air, "metal_tuple_elementwise_air"));
    parameter_numbers.push_back(emitter.parameter_numbers());
    num_elements.push_back(emitter.num_work_items());
    metallibs.push_back(std::move(metallib));
  }

  return std::make_unique<MetalTupleElementwiseExecutable>(
      std::move(module), std::move(parameter_numbers), std::move(num_elements),
      std::move(metallibs));
}

absl::StatusOr<std::unique_ptr<Executable>> BuildMetalElementwiseExecutable(
    std::shared_ptr<HloModule> module) {
  HloInstruction* root = module->entry_computation()->root_instruction();
  if (root->opcode() == HloOpcode::kTuple || root->shape().IsTuple()) {
    return BuildMetalTupleElementwiseExecutable(std::move(module));
  }
  ElementwiseAirEmitter emitter(root->shape());
  TF_ASSIGN_OR_RETURN(std::string air, emitter.Emit(root));
  TF_ASSIGN_OR_RETURN(std::vector<uint8_t> metallib,
                      CompileMetalAirToMetallib(air, "metal_elementwise_air"));
  return std::make_unique<MetalElementwiseExecutable>(
      std::move(module), emitter.parameter_numbers(), emitter.num_work_items(),
      std::move(metallib));
}

absl::StatusOr<const HloInstruction*> MatchTopKTupleRoot(
    const HloInstruction* root) {
  if (root->opcode() == HloOpcode::kTopK) {
    return root;
  }
  if (root->opcode() != HloOpcode::kTuple || root->operand_count() != 2) {
    return absl::UnimplementedError(
        "Metal direct AIR topk tuple executable requires a topk root or a "
        "tuple of topk get-tuple-elements.");
  }

  const HloInstruction* values = root->operand(0);
  const HloInstruction* indices = root->operand(1);
  if (values->opcode() != HloOpcode::kGetTupleElement ||
      indices->opcode() != HloOpcode::kGetTupleElement ||
      values->tuple_index() != 0 || indices->tuple_index() != 1 ||
      values->operand(0) != indices->operand(0) ||
      values->operand(0)->opcode() != HloOpcode::kTopK) {
    return absl::UnimplementedError(
        "Metal direct AIR topk tuple executable requires tuple(gte(topk, 0), "
        "gte(topk, 1)).");
  }
  return values->operand(0);
}

class MetalTopKTupleExecutable final : public Executable {
 public:
  MetalTopKTupleExecutable(std::shared_ptr<HloModule> module,
                           std::vector<int64_t> values_parameter_numbers,
                           std::vector<int64_t> indices_parameter_numbers,
                           std::vector<uint8_t> values_metallib,
                           std::vector<uint8_t> indices_metallib)
      : Executable(std::move(module)),
        values_parameter_numbers_(std::move(values_parameter_numbers)),
        indices_parameter_numbers_(std::move(indices_parameter_numbers)),
        result_shape_(this->module().output_shape()),
        num_elements_(ShapeUtil::ElementsIn(result_shape_.tuple_shapes(0))),
        values_metallib_(std::move(values_metallib)),
        indices_metallib_(std::move(indices_metallib)) {}

  Shape result_shape() const override { return result_shape_; }

  absl::StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments) override {
    se::Stream* stream = run_options->stream();
    if (stream == nullptr) {
      return absl::InvalidArgumentError("Metal topk tuple requires a stream.");
    }
    se::StreamExecutor* executor = stream->parent();
    se::DeviceAddressAllocator* allocator = run_options->allocator();
    if (allocator == nullptr) {
      return absl::InvalidArgumentError(
          "Metal topk tuple requires an allocator.");
    }
    TF_ASSIGN_OR_RETURN(TransferManager * transfer_manager,
                        TransferManager::GetForPlatform(executor->GetPlatform()));

    const int device_ordinal = run_options->device_ordinal() != -1
                                   ? run_options->device_ordinal()
                                   : executor->device_ordinal();
    const Shape& values_shape = result_shape_.tuple_shapes(0);
    const Shape& indices_shape = result_shape_.tuple_shapes(1);

    TF_ASSIGN_OR_RETURN(
        se::ScopedDeviceAddress<uint8_t> tuple_buffer,
        allocator->Allocate(device_ordinal,
                            transfer_manager->GetByteSizeRequirement(
                                result_shape_)));
    TF_ASSIGN_OR_RETURN(
        se::ScopedDeviceAddress<uint8_t> values_buffer,
        allocator->Allocate(device_ordinal,
                            transfer_manager->GetByteSizeRequirement(
                                values_shape)));
    TF_ASSIGN_OR_RETURN(
        se::ScopedDeviceAddress<uint8_t> indices_buffer,
        allocator->Allocate(device_ordinal,
                            transfer_manager->GetByteSizeRequirement(
                                indices_shape)));
    se::DeviceAddressBase values_output = *values_buffer;
    se::DeviceAddressBase indices_output = *indices_buffer;

    TF_ASSIGN_OR_RETURN(se::ScopedDeviceAddress<uint8_t> params_buffer,
                        allocator->Allocate(device_ordinal,
                                            sizeof(ElementwiseParams)));
    se::DeviceAddressBase params_address = *params_buffer;
    ElementwiseParams params{static_cast<uint32_t>(num_elements_), 0, 0, 0};
    TF_RETURN_IF_ERROR(
        stream->Memcpy(&params_address, &params, sizeof(ElementwiseParams)));

    TF_RETURN_IF_ERROR(LaunchElementwiseKernel(
        executor, stream, values_metallib_, values_parameter_numbers_,
        arguments, values_output, params_address, "topk values"));
    TF_RETURN_IF_ERROR(LaunchElementwiseKernel(
        executor, stream, indices_metallib_, indices_parameter_numbers_,
        arguments, indices_output, params_address, "topk indices"));

    ExecutionOutput result(result_shape_, allocator, device_ordinal,
                           executor->device_ordinal());
    ScopedShapedBuffer* shaped_result = result.MutableResult();
    ShapedBuffer* shaped_buffer = static_cast<ShapedBuffer*>(shaped_result);
    shaped_buffer->set_buffer(tuple_buffer.Release(), {});
    shaped_buffer->set_buffer(values_buffer.Release(), {0});
    shaped_buffer->set_buffer(indices_buffer.Release(), {1});
    TF_RETURN_IF_ERROR(
        transfer_manager->WriteRootTupleIndexTable(stream, *shaped_result));
    TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());

    result.Commit();
    return std::move(result);
  }

 private:
  absl::Status LaunchElementwiseKernel(
      se::StreamExecutor* executor, se::Stream* stream,
      const std::vector<uint8_t>& metallib,
      const std::vector<int64_t>& parameter_numbers,
      const std::vector<ExecutionInput>& arguments, se::DeviceAddressBase output,
      se::DeviceAddressBase params_address, absl::string_view name) const {
    auto spec = se::KernelLoaderSpec::CreateOwningMetalLibraryInMemorySpec(
        std::vector<uint8_t>(metallib.begin(), metallib.end()),
        "elementwise_f32", parameter_numbers.size() + 2);
    TF_ASSIGN_OR_RETURN(std::unique_ptr<se::Kernel> kernel,
                        executor->LoadKernel(spec));

    se::KernelArgsPackedArray kernel_args(parameter_numbers.size() + 2);
    for (int64_t parameter_number : parameter_numbers) {
      if (parameter_number < 0 || parameter_number >= arguments.size()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Metal %s parameter number is out of bounds.", name));
      }
      se::DeviceAddressBase arg =
          arguments[parameter_number].Buffer({}).AsDeviceAddress();
      if (arg.is_null()) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Metal %s input buffer is null.", name));
      }
      kernel_args.add_argument(arg);
    }
    kernel_args.add_argument(output);
    kernel_args.add_argument(params_address);

    se::ThreadDim threads(/*x=*/256, /*y=*/1, /*z=*/1);
    se::BlockDim blocks(
        /*x=*/static_cast<uint64_t>((num_elements_ + 255) / 256),
        /*y=*/1, /*z=*/1);
    return kernel->Launch(threads, blocks, stream, kernel_args);
  }

  std::vector<int64_t> values_parameter_numbers_;
  std::vector<int64_t> indices_parameter_numbers_;
  Shape result_shape_;
  int64_t num_elements_;
  std::vector<uint8_t> values_metallib_;
  std::vector<uint8_t> indices_metallib_;
};

absl::StatusOr<std::unique_ptr<Executable>> BuildMetalTopKTupleExecutable(
    std::shared_ptr<HloModule> module) {
  HloInstruction* root = module->entry_computation()->root_instruction();
  TF_ASSIGN_OR_RETURN(const HloInstruction* topk, MatchTopKTupleRoot(root));
  if (!root->shape().IsTuple() || !topk->shape().IsTuple() ||
      ShapeUtil::TupleElementCount(root->shape()) != 2 ||
      ShapeUtil::TupleElementCount(topk->shape()) != 2 ||
      !ShapeUtil::Equal(root->shape().tuple_shapes(0),
                        topk->shape().tuple_shapes(0)) ||
      !ShapeUtil::Equal(root->shape().tuple_shapes(1),
                        topk->shape().tuple_shapes(1)) ||
      (!IsF32Array(root->shape().tuple_shapes(0)) &&
       !IsS32Array(root->shape().tuple_shapes(0))) ||
      !IsS32Array(root->shape().tuple_shapes(1)) ||
      root->shape().tuple_shapes(0).dimensions() !=
          root->shape().tuple_shapes(1).dimensions()) {
    return absl::UnimplementedError(
        "Metal direct AIR topk tuple executable supports only tuple roots "
        "with f32 or s32 values and s32 indices of the same shape.");
  }

  ElementwiseAirEmitter values_emitter(root->shape().tuple_shapes(0));
  TF_ASSIGN_OR_RETURN(std::string values_air,
                      values_emitter.EmitTopKTupleElement(topk, 0));
  ElementwiseAirEmitter indices_emitter(root->shape().tuple_shapes(1));
  TF_ASSIGN_OR_RETURN(std::string indices_air,
                      indices_emitter.EmitTopKTupleElement(topk, 1));

  TF_ASSIGN_OR_RETURN(
      std::vector<uint8_t> values_metallib,
      CompileMetalAirToMetallib(values_air, "metal_topk_values_air"));
  TF_ASSIGN_OR_RETURN(
      std::vector<uint8_t> indices_metallib,
      CompileMetalAirToMetallib(indices_air, "metal_topk_indices_air"));

  return std::make_unique<MetalTopKTupleExecutable>(
      std::move(module), values_emitter.parameter_numbers(),
      indices_emitter.parameter_numbers(), std::move(values_metallib),
      std::move(indices_metallib));
}

MetalMatmulExecutable::MetalMatmulExecutable(std::shared_ptr<HloModule> module,
                                             MetalMatmulConfig config,
                                             std::vector<uint8_t> metallib)
    : Executable(std::move(module)),
      config_(config),
      result_shape_(this->module().output_shape()),
      kernel_name_(config.use_simdgroup
                       ? (config.relu ? "matmul_relu_simdgroup_8x8"
                                      : "matmul_simdgroup_8x8")
                       : (config.relu ? "matmul_relu_scalar"
                                      : "matmul_scalar")),
      metallib_(std::move(metallib)) {}

Shape MetalMatmulExecutable::result_shape() const { return result_shape_; }

absl::StatusOr<ExecutionOutput> MetalMatmulExecutable::ExecuteAsyncOnStream(
    const ServiceExecutableRunOptions* run_options,
    std::vector<ExecutionInput> arguments) {
  if (arguments.size() != 2) {
    return absl::InvalidArgumentError(
        "Metal matmul executable expects exactly two arguments.");
  }

  se::Stream* stream = run_options->stream();
  if (stream == nullptr) {
    return absl::InvalidArgumentError("Metal matmul requires a stream.");
  }
  se::StreamExecutor* executor = stream->parent();
  se::DeviceAddressAllocator* allocator = run_options->allocator();
  if (allocator == nullptr) {
    return absl::InvalidArgumentError("Metal matmul requires an allocator.");
  }

  const int device_ordinal = run_options->device_ordinal() != -1
                                 ? run_options->device_ordinal()
                                 : executor->device_ordinal();

  se::DeviceAddressBase lhs = arguments[0].Buffer({}).AsDeviceAddress();
  se::DeviceAddressBase rhs = arguments[1].Buffer({}).AsDeviceAddress();
  if (lhs.is_null() || rhs.is_null()) {
    return absl::InvalidArgumentError("Metal matmul input buffer is null.");
  }

  const int64_t output_size = ShapeUtil::ByteSizeOf(result_shape_, 8);
  TF_ASSIGN_OR_RETURN(se::ScopedDeviceAddress<uint8_t> output_buffer,
                      allocator->Allocate(device_ordinal, output_size));
  se::DeviceAddressBase output = *output_buffer;

  TF_ASSIGN_OR_RETURN(se::ScopedDeviceAddress<uint8_t> params_buffer,
                      allocator->Allocate(device_ordinal, sizeof(MatmulParams)));
  se::DeviceAddressBase params_address = *params_buffer;
  MatmulParams params{static_cast<uint32_t>(config_.m),
                      static_cast<uint32_t>(config_.n),
                      static_cast<uint32_t>(config_.k), 0};
  TF_RETURN_IF_ERROR(
      stream->Memcpy(&params_address, &params, sizeof(MatmulParams)));

  auto spec = se::KernelLoaderSpec::CreateOwningMetalLibraryInMemorySpec(
      std::vector<uint8_t>(metallib_.begin(), metallib_.end()), kernel_name_,
      /*arity=*/4);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<se::Kernel> kernel,
                      executor->LoadKernel(spec));

  se::KernelArgsPackedArray kernel_args(/*num_args=*/4);
  kernel_args.add_argument(lhs);
  kernel_args.add_argument(rhs);
  kernel_args.add_argument(output);
  kernel_args.add_argument(params_address);

  se::ThreadDim threads(/*x=*/256, /*y=*/1, /*z=*/1);
  se::BlockDim blocks(
      /*x=*/config_.use_simdgroup
          ? static_cast<uint64_t>((config_.n + 31) / 32)
          : static_cast<uint64_t>((config_.m * config_.n + 255) / 256),
      /*y=*/config_.use_simdgroup
          ? static_cast<uint64_t>((config_.m + 15) / 16)
          : 1,
      /*z=*/1);
  TF_RETURN_IF_ERROR(kernel->Launch(threads, blocks, stream, kernel_args));
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());

  ExecutionOutput result(result_shape_, allocator, device_ordinal,
                         executor->device_ordinal());
  static_cast<ShapedBuffer*>(result.MutableResult())
      ->set_buffer(output_buffer.Release(), {});
  result.Commit();
  return std::move(result);
}

}  // namespace gpu
}  // namespace xla
