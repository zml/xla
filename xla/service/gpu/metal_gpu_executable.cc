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

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/embed_metal_air_kernels.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
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
  uint32_t num_elements;
  uint32_t reserved0;
  uint32_t reserved1;
  uint32_t reserved2;
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
};

struct MetalReductionConfig {
  int64_t parameter_number = 0;
  int64_t num_elements = 0;
  int64_t input_start = 0;
  int64_t input_stride = 1;
  float init_value = 0.0f;
  float output_scale = 1.0f;
  ReductionKind kind = ReductionKind::kAdd;
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

bool IsF32Array(const Shape& shape) {
  return shape.element_type() == F32 && shape.IsArray();
}

bool IsPredArray(const Shape& shape) {
  return shape.element_type() == PRED && shape.IsArray();
}

bool IsS32Array(const Shape& shape) {
  return shape.element_type() == S32 && shape.IsArray();
}

bool IsScalarLikeF32(const Shape& shape) {
  return shape.element_type() == F32 && ShapeUtil::IsEffectiveScalar(shape);
}

std::string FloatLiteral(float value) {
  double double_value = static_cast<double>(value);
  uint64_t bits = 0;
  std::memcpy(&bits, &double_value, sizeof(bits));
  return absl::StrFormat("0x%016llX",
                         static_cast<unsigned long long>(bits));
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
      : result_shape_(std::move(result_shape)) {}

  absl::StatusOr<std::string> Emit(const HloInstruction* root) {
    if (!IsF32Array(root->shape()) && !IsPredArray(root->shape())) {
      return absl::UnimplementedError(
          "Metal direct AIR elementwise supports only f32 and pred arrays.");
    }
    std::vector<std::string> body;
    TF_ASSIGN_OR_RETURN(std::string value,
                        EmitValue(root, /*force_scalar=*/false, &body));
    expression_body_ = std::move(body);
    result_value_ = std::move(value);
    return BuildModule();
  }

  const std::vector<int64_t>& parameter_numbers() const {
    return parameter_numbers_;
  }

 private:
  absl::StatusOr<std::string> EmitValue(const HloInstruction* instr,
                                        bool force_scalar,
                                        std::vector<std::string>* body) {
    if (instr->opcode() == HloOpcode::kBroadcast) {
      if (!IsScalarLikeF32(instr->operand(0)->shape())) {
        return absl::UnimplementedError(
            "Metal direct AIR elementwise currently supports only scalar "
            "broadcasts.");
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
      if (!IsScalarLikeF32(instr->shape())) {
        return absl::UnimplementedError(
            "Metal direct AIR elementwise currently supports only scalar "
            "constants.");
      }
      return FloatLiteral(instr->literal().Get<float>({}));
    }

    if (instr->opcode() == HloOpcode::kParameter) {
      if (!force_scalar && !HasResultDimensions(instr->shape())) {
        return absl::UnimplementedError(
          "Metal direct AIR elementwise parameters must have the result "
          "shape unless they are scalar-broadcast operands.");
      }
      const int64_t parameter_number = instr->parameter_number();
      const int input_index = InputIndexForParameter(parameter_number);
      return EmitLoad(input_index, force_scalar ? "0" : "%idx", body);
    }

    switch (instr->opcode()) {
      case HloOpcode::kCall:
        return EmitCall(instr, body);
      case HloOpcode::kConcatenate:
        return EmitConcatenate(instr, body);
      case HloOpcode::kIota:
        return EmitIota(instr, body);
      case HloOpcode::kTranspose:
        return EmitTranspose(instr, body);
      case HloOpcode::kCompare:
        return EmitCompare(instr, body);
      case HloOpcode::kSelect:
        return EmitSelect(instr, body);
      case HloOpcode::kAdd:
      case HloOpcode::kSubtract:
      case HloOpcode::kMultiply:
      case HloOpcode::kDivide:
      case HloOpcode::kMaximum:
      case HloOpcode::kMinimum:
        return EmitBinary(instr, body);
      case HloOpcode::kNegate:
        return EmitUnary(instr, "fneg fast float", body);
      case HloOpcode::kAbs:
        return EmitAbs(instr, body);
      case HloOpcode::kCos:
        return EmitIntrinsicUnary(instr, "air.fast_cos.f32", body);
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
      case HloOpcode::kSqrt:
        return EmitIntrinsicUnary(instr, "air.fast_sqrt.f32", body);
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
        return EmitIntrinsicUnary(instr, "air.sign.f32", body);
      default:
        return absl::UnimplementedError(absl::StrFormat(
            "Metal direct AIR elementwise does not support HLO opcode %s.",
            HloOpcodeString(instr->opcode())));
    }
  }

  absl::StatusOr<std::string> EmitLoadFromLinearIndex(
      const HloInstruction* instr, absl::string_view index,
      std::vector<std::string>* body) {
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
    if (instr->opcode() == HloOpcode::kSlice) {
      if (!IsF32Array(instr->shape()) ||
          !IsF32Array(instr->operand(0)->shape()) ||
          instr->shape().dimensions().size() != 1 ||
          instr->operand(0)->shape().dimensions().size() != 1 ||
          instr->slice_strides().size() != 1) {
        return absl::UnimplementedError(
            "Metal direct AIR linear load slice currently supports only "
            "rank-1 f32 slices.");
      }
      std::string source_index = std::string(index);
      const int64_t stride = instr->slice_strides(0);
      const int64_t start = instr->slice_starts(0);
      if (stride != 1) {
        std::string scaled = NewName("linear_slice_scaled");
        body->push_back(absl::StrFormat("  %s = mul i64 %s, %d", scaled,
                                        source_index, stride));
        source_index = scaled;
      }
      if (start != 0) {
        std::string shifted = NewName("linear_slice_idx");
        body->push_back(absl::StrFormat("  %s = add i64 %s, %d", shifted,
                                        source_index, start));
        source_index = shifted;
      }
      return EmitLoadFromLinearIndex(instr->operand(0), source_index, body);
    }
    if (instr->opcode() == HloOpcode::kParameter) {
      const int input_index = InputIndexForParameter(instr->parameter_number());
      return EmitLoad(input_index, index, body);
    }
    return absl::UnimplementedError(absl::StrFormat(
        "Metal direct AIR linear load does not support HLO opcode %s.",
        HloOpcodeString(instr->opcode())));
  }

  absl::StatusOr<std::string> EmitTranspose(const HloInstruction* instr,
                                            std::vector<std::string>* body) {
    const HloInstruction* operand = instr->operand(0);
    if (!IsF32Array(instr->shape()) || !IsF32Array(operand->shape()) ||
        instr->shape().dimensions().size() != 2 ||
        operand->shape().dimensions().size() != 2 ||
        instr->dimensions().size() != 2 || instr->dimensions()[0] != 1 ||
        instr->dimensions()[1] != 0) {
      return absl::UnimplementedError(
          "Metal direct AIR elementwise transpose currently supports only "
          "rank-2 f32 transposes with permutation {1,0}.");
    }
    return EmitLoadFromLinearIndex(operand, "%idx", body);
  }

  absl::StatusOr<std::string> EmitConcatenate(const HloInstruction* instr,
                                              std::vector<std::string>* body) {
    if (!IsF32Array(instr->shape()) ||
        instr->shape().dimensions().size() != 1 ||
        instr->dimensions().size() != 1 || instr->dimensions()[0] != 0 ||
        instr->operand_count() == 0) {
      return absl::UnimplementedError(
          "Metal direct AIR elementwise concatenate currently supports only "
          "rank-1 f32 concatenates along dimension 0.");
    }

    const HloInstruction* parameter = nullptr;
    int64_t base_start = 0;
    int64_t output_offset = 0;
    for (const HloInstruction* operand : instr->operands()) {
      if (!IsF32Array(operand->shape()) ||
          operand->shape().dimensions().size() != 1) {
        return absl::UnimplementedError(
            "Metal direct AIR elementwise concatenate operands must be rank-1 "
            "f32 arrays.");
      }
      const HloInstruction* operand_parameter = operand;
      int64_t operand_start = 0;
      if (operand->opcode() == HloOpcode::kSlice) {
        if (operand->slice_strides().size() != 1 ||
            operand->slice_strides(0) != 1 ||
            operand->operand(0)->opcode() != HloOpcode::kParameter) {
          return absl::UnimplementedError(
              "Metal direct AIR elementwise concatenate currently supports "
              "only contiguous slices of parameters.");
        }
        operand_parameter = operand->operand(0);
        operand_start = operand->slice_starts(0);
      }
      if (operand_parameter->opcode() != HloOpcode::kParameter) {
        return absl::UnimplementedError(
            "Metal direct AIR elementwise concatenate currently supports only "
            "parameter or slice operands.");
      }
      if (parameter == nullptr) {
        parameter = operand_parameter;
        base_start = operand_start;
      }
      if (operand_parameter != parameter ||
          operand_start != base_start + output_offset) {
        return absl::UnimplementedError(
            "Metal direct AIR elementwise concatenate currently supports only "
            "contiguous views of one parameter.");
      }
      output_offset += ShapeUtil::ElementsIn(operand->shape());
    }

    std::string source_index = "%idx";
    if (base_start != 0) {
      source_index = NewName("concat_idx");
      body->push_back(absl::StrFormat("  %s = add i64 %%idx, %d",
                                      source_index, base_start));
    }
    return EmitLoadFromLinearIndex(parameter, source_index, body);
  }

  absl::StatusOr<std::string> EmitSlice(const HloInstruction* instr,
                                        bool force_scalar,
                                        std::vector<std::string>* body) {
    if (!IsF32Array(instr->shape()) ||
        !IsF32Array(instr->operand(0)->shape()) ||
        instr->shape().dimensions().size() != 1 ||
        instr->operand(0)->shape().dimensions().size() != 1 ||
        instr->slice_strides().size() != 1) {
      return absl::UnimplementedError(
          "Metal direct AIR elementwise slice currently supports only rank-1 "
          "f32 slices.");
    }
    const int64_t slice_elements = ShapeUtil::ElementsIn(instr->shape());
    if ((!force_scalar &&
         slice_elements != ShapeUtil::ElementsIn(result_shape_)) ||
        (force_scalar && slice_elements != 1)) {
      return absl::UnimplementedError(
          "Metal direct AIR elementwise slice must have the same element count "
          "as the final result, unless used as a scalar operand.");
    }

    const HloInstruction* operand = instr->operand(0);
    while (operand->opcode() == HloOpcode::kBitcast ||
           operand->opcode() == HloOpcode::kReshape) {
      operand = operand->operand(0);
    }
    if (operand->opcode() != HloOpcode::kParameter) {
      return absl::UnimplementedError(
          "Metal direct AIR elementwise slice currently supports only "
          "parameter operands.");
    }
    const int input_index =
        InputIndexForParameter(operand->parameter_number());
    const int64_t start = instr->slice_starts(0);
    const int64_t stride = instr->slice_strides(0);
    std::string source_index = force_scalar ? "0" : "%idx";
    if (stride != 1) {
      std::string scaled = NewName("slice_scaled");
      body->push_back(absl::StrFormat("  %s = mul i64 %%idx, %d", scaled,
                                      stride));
      source_index = scaled;
    }
    if (start != 0) {
      std::string shifted = NewName("slice_idx");
      body->push_back(absl::StrFormat("  %s = add i64 %s, %d", shifted,
                                      source_index, start));
      source_index = shifted;
    }
    return EmitLoad(input_index, source_index, body);
  }

  absl::StatusOr<std::string> EmitIota(const HloInstruction* instr,
                                       std::vector<std::string>* body) {
    if (!IsF32Array(instr->shape()) ||
        !ShapeUtil::Equal(instr->shape(), result_shape_) ||
        instr->shape().dimensions().size() != 1) {
      return absl::UnimplementedError(
          "Metal direct AIR elementwise iota currently supports only rank-1 "
          "f32 iota results over dimension 0.");
    }
    std::string value = NewName("iota");
    body->push_back(
        absl::StrFormat("  %s = uitofp i64 %%idx to float", value));
    return value;
  }

  int InputIndexForParameter(int64_t parameter_number) {
    auto it = parameter_to_input_index_.find(parameter_number);
    if (it != parameter_to_input_index_.end()) {
      return it->second;
    }
    const int input_index = parameter_numbers_.size();
    parameter_to_input_index_[parameter_number] = input_index;
    parameter_numbers_.push_back(parameter_number);
    return input_index;
  }

  absl::StatusOr<const HloInstruction*> ResolveCallParameter(
      const HloInstruction* call, const HloInstruction* callee_operand) {
    if (callee_operand->opcode() != HloOpcode::kParameter) {
      return absl::UnimplementedError(
          "Metal direct AIR elementwise call inlining currently supports only "
          "callee roots whose operands are parameters.");
    }
    const int64_t parameter_number = callee_operand->parameter_number();
    if (parameter_number < 0 || parameter_number >= call->operand_count()) {
      return absl::InvalidArgumentError(
          "Metal direct AIR elementwise call parameter is out of bounds.");
    }
    return call->operand(parameter_number);
  }

  absl::StatusOr<std::string> EmitCall(const HloInstruction* instr,
                                       std::vector<std::string>* body) {
    const HloInstruction* root = instr->to_apply()->root_instruction();
    if (root->opcode() != HloOpcode::kSelect) {
      return absl::UnimplementedError(absl::StrFormat(
          "Metal direct AIR elementwise supports only calls that inline to "
          "select, got callee root opcode %s.",
          HloOpcodeString(root->opcode())));
    }
    TF_ASSIGN_OR_RETURN(const HloInstruction* pred_instr,
                        ResolveCallParameter(instr, root->operand(0)));
    TF_ASSIGN_OR_RETURN(const HloInstruction* on_true_instr,
                        ResolveCallParameter(instr, root->operand(1)));
    TF_ASSIGN_OR_RETURN(const HloInstruction* on_false_instr,
                        ResolveCallParameter(instr, root->operand(2)));

    TF_ASSIGN_OR_RETURN(std::string pred,
                        EmitValue(pred_instr, /*force_scalar=*/false, body));
    TF_ASSIGN_OR_RETURN(
        std::string on_true,
        EmitValue(on_true_instr, IsScalarLikeF32(on_true_instr->shape()), body));
    TF_ASSIGN_OR_RETURN(std::string on_false,
                        EmitValue(on_false_instr,
                                  IsScalarLikeF32(on_false_instr->shape()),
                                  body));
    std::string value = NewName("select");
    body->push_back(absl::StrFormat("  %s = select i1 %s, float %s, float %s",
                                    value, pred, on_true, on_false));
    return value;
  }

  absl::StatusOr<std::string> EmitCompare(const HloInstruction* instr,
                                          std::vector<std::string>* body) {
    Shape pred_shape = ShapeUtil::MakeShape(PRED, result_shape_.dimensions());
    if (!ShapeUtil::Compatible(instr->shape(), pred_shape)) {
      return absl::UnimplementedError(
          "Metal direct AIR elementwise compare currently supports only "
          "predicates matching the result shape.");
    }
    TF_ASSIGN_OR_RETURN(std::string lhs,
                        EmitValue(instr->operand(0), IsScalarOperand(instr, 0),
                                  body));
    TF_ASSIGN_OR_RETURN(std::string rhs,
                        EmitValue(instr->operand(1), IsScalarOperand(instr, 1),
                                  body));
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
    std::string cmp = NewName("cmp");
    body->push_back(absl::StrFormat("  %s = fcmp fast %s float %s, %s", cmp,
                                    predicate, lhs, rhs));
    return cmp;
  }

  absl::StatusOr<std::string> EmitSelect(const HloInstruction* instr,
                                         std::vector<std::string>* body) {
    if (!IsF32Array(instr->shape())) {
      return absl::UnimplementedError(
          "Metal direct AIR elementwise select currently supports only f32 "
          "array results.");
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
    body->push_back(absl::StrFormat("  %s = select i1 %s, float %s, float %s",
                                    value, pred, on_true, on_false));
    return value;
  }

  absl::StatusOr<std::string> EmitBinary(const HloInstruction* instr,
                                         std::vector<std::string>* body) {
    TF_ASSIGN_OR_RETURN(std::string lhs,
                        EmitValue(instr->operand(0), IsScalarOperand(instr, 0),
                                  body));
    TF_ASSIGN_OR_RETURN(std::string rhs,
                        EmitValue(instr->operand(1), IsScalarOperand(instr, 1),
                                  body));
    switch (instr->opcode()) {
      case HloOpcode::kAdd:
        return EmitOp("fadd fast float", lhs, rhs, body);
      case HloOpcode::kSubtract:
        return EmitOp("fsub fast float", lhs, rhs, body);
      case HloOpcode::kMultiply:
        return EmitOp("fmul fast float", lhs, rhs, body);
      case HloOpcode::kDivide:
        return EmitOp("fdiv fast float", lhs, rhs, body);
      case HloOpcode::kMaximum:
        return EmitCompareSelect("ogt", lhs, rhs, body);
      case HloOpcode::kMinimum:
        return EmitCompareSelect("olt", lhs, rhs, body);
      default:
        return absl::InternalError("Unexpected binary HLO opcode.");
    }
  }

  absl::StatusOr<std::string> EmitUnary(const HloInstruction* instr,
                                        absl::string_view op,
                                        std::vector<std::string>* body) {
    TF_ASSIGN_OR_RETURN(std::string value,
                        EmitValue(instr->operand(0), IsScalarOperand(instr, 0),
                                  body));
    std::string name = NewName("unary");
    body->push_back(absl::StrFormat("  %s = %s %s", name, op, value));
    return name;
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

  absl::StatusOr<std::string> EmitAbs(const HloInstruction* instr,
                                      std::vector<std::string>* body) {
    TF_ASSIGN_OR_RETURN(std::string value,
                        EmitValue(instr->operand(0), IsScalarOperand(instr, 0),
                                  body));
    std::string neg = NewName("neg");
    body->push_back(absl::StrFormat("  %s = fneg fast float %s", neg, value));
    return EmitCompareSelect("ogt", value, neg, body);
  }

  bool IsScalarOperand(const HloInstruction* instr, int operand_index) const {
    return IsScalarLikeF32(instr->operand(operand_index)->shape());
  }

  bool HasResultDimensions(const Shape& shape) const {
    return shape.IsArray() && shape.dimensions() == result_shape_.dimensions();
  }

  std::string EmitLoad(int input_index, absl::string_view index,
                       std::vector<std::string>* body) {
    std::string ptr = NewName("ptr");
    std::string value = NewName("value");
    body->push_back(absl::StrFormat(
        "  %s = getelementptr inbounds float, float addrspace(1)* %%arg%d, "
        "i64 %s",
        ptr, input_index, index));
    body->push_back(absl::StrFormat(
        "  %s = load float, float addrspace(1)* %s, align 4", value, ptr));
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

  std::string NewName(absl::string_view prefix) {
    return absl::StrFormat("%%%s%d", prefix, next_value_id_++);
  }

  std::string BuildModule() const {
    const bool result_is_pred = result_shape_.element_type() == PRED;
    std::vector<std::string> args;
    for (int i = 0; i < parameter_numbers_.size(); ++i) {
      args.push_back(absl::StrFormat(
          "    float addrspace(1)* nocapture noundef readonly "
          "\"air-buffer-no-alias\" %%arg%d",
          i));
    }
    args.push_back(absl::StrFormat(
        "    %s addrspace(1)* nocapture noundef writeonly "
        "\"air-buffer-no-alias\" %%out",
        result_is_pred ? "i8" : "float"));
    args.push_back(
        "    %struct.ElementwiseParams addrspace(2)* nocapture noundef "
        "readonly align 4 dereferenceable(16) \"air-buffer-no-alias\" %params");
    args.push_back("    <3 x i32> noundef %gid");

    std::vector<std::string> signature_types(parameter_numbers_.size(),
                                             "float addrspace(1)*");
    signature_types.push_back(result_is_pred ? "i8 addrspace(1)*"
                                             : "float addrspace(1)*");
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

    std::string store_result;
    if (result_is_pred) {
      store_result = absl::StrFormat(R"(  %%out_ptr = getelementptr inbounds i8, i8 addrspace(1)* %%out, i64 %%idx
  %%out_i8 = zext i1 %s to i8
  store i8 %%out_i8, i8 addrspace(1)* %%out_ptr, align 1)",
                                     result_value_);
    } else {
      store_result = absl::StrFormat(R"(  %%out_ptr = getelementptr inbounds float, float addrspace(1)* %%out, i64 %%idx
  store float %s, float addrspace(1)* %%out_ptr, align 4)",
                                     result_value_);
    }

    std::string module = absl::StrFormat(R"(
source_filename = "xla/service/gpu/metal_elementwise_air"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024-n8:16:32"
target triple = "air64_v27-apple-macosx15.0.0"

%%struct.ElementwiseParams = type { i32, i32, i32, i32 }

declare float @air.fast_cos.f32(float) local_unnamed_addr #1
declare float @air.fast_ceil.f32(float) local_unnamed_addr #1
declare float @air.fast_exp.f32(float) local_unnamed_addr #1
declare float @air.fast_floor.f32(float) local_unnamed_addr #1
declare float @air.fast_log.f32(float) local_unnamed_addr #1
declare float @air.fast_rint.f32(float) local_unnamed_addr #1
declare float @air.fast_round.f32(float) local_unnamed_addr #1
declare float @air.fast_rsqrt.f32(float) local_unnamed_addr #1
declare float @air.fast_sin.f32(float) local_unnamed_addr #1
declare float @air.fast_sqrt.f32(float) local_unnamed_addr #1
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
!llvm.module.flags = !{!10, !11, !12, !13, !14, !15, !16, !17}
!air.compile_options = !{!18, !19, !20}
!llvm.ident = !{!21}
!air.version = !{!22}
!air.language_version = !{!23}
!air.source_file_name = !{!24}

!0 = !{void (%s)* @elementwise_f32, !1, !2}
!1 = !{}
!2 = !{%s}
)",
                                         absl::StrJoin(args, ",\n"),
                                         absl::StrJoin(expression_body_, "\n"),
                                         store_result,
                                         absl::StrJoin(signature_types, ", "),
                                         absl::StrJoin(metadata_args, ", "));

    for (int i = 0; i < parameter_numbers_.size(); ++i) {
      absl::StrAppendFormat(
          &module,
          "!%d = !{i32 %d, !\"air.buffer\", !\"air.location_index\", i32 %d, "
          "i32 1, !\"air.read\", !\"air.address_space\", i32 1, "
          "!\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 "
          "4, !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", "
          "!\"arg%d\"}\n",
          3 + i, i, i, i);
    }
    absl::StrAppendFormat(
        &module,
        "!%d = !{i32 %d, !\"air.buffer\", !\"air.location_index\", i32 %d, "
        "i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, "
        "!\"air.arg_type_size\", i32 %d, !\"air.arg_type_align_size\", i32 %d, "
        "!\"air.arg_type_name\", !\"%s\", !\"air.arg_name\", !\"out\"}\n",
        output_metadata, static_cast<int>(parameter_numbers_.size()),
        static_cast<int>(parameter_numbers_.size()), result_is_pred ? 1 : 4,
        result_is_pred ? 1 : 4, result_is_pred ? "bool" : "float");
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
    absl::StrAppend(&module, R"(
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
!21 = !{!"xla direct AIR elementwise"}
!22 = !{i32 2, i32 7, i32 0}
!23 = !{!"Metal", i32 3, i32 2, i32 0}
!24 = !{!"xla/service/gpu/metal_elementwise_air"}
)");
    return module;
  }

  Shape result_shape_;
  absl::flat_hash_map<int64_t, int> parameter_to_input_index_;
  std::vector<int64_t> parameter_numbers_;
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
  if (!IsF32Rank2Array(lhs->shape()) || !IsF32Rank2Array(rhs->shape()) ||
      !IsF32Rank2Array(dot->shape())) {
    return absl::UnimplementedError(
        "Metal direct AIR matmul supports only rank-2 f32 arrays.");
  }

  const DotDimensionNumbers& dims = dot->dot_dimension_numbers();
  if (dims.lhs_batch_dimensions_size() != 0 ||
      dims.rhs_batch_dimensions_size() != 0 ||
      dims.lhs_contracting_dimensions_size() != 1 ||
      dims.rhs_contracting_dimensions_size() != 1 ||
      dims.lhs_contracting_dimensions(0) != 1 ||
      dims.rhs_contracting_dimensions(0) != 0) {
    return absl::UnimplementedError(
        "Metal direct AIR matmul supports only non-batched row-major "
        "contracting dimensions lhs[1] x rhs[0].");
  }

  MetalMatmulConfig config;
  config.m = lhs->shape().dimensions(0);
  config.k = lhs->shape().dimensions(1);
  config.n = rhs->shape().dimensions(1);
  config.relu = relu;

  if (rhs->shape().dimensions(0) != config.k ||
      dot->shape().dimensions(0) != config.m ||
      dot->shape().dimensions(1) != config.n) {
    return absl::InvalidArgumentError(
        "Metal direct AIR matmul shape dimensions are inconsistent.");
  }

  if (config.m % 16 != 0 || config.n % 32 != 0 || config.k % 8 != 0) {
    return absl::UnimplementedError(
        "Metal direct AIR simdgroup matmul currently requires M multiple of "
        "16, N multiple of 32, and K multiple of 8.");
  }

  return config;
}

absl::StatusOr<std::vector<uint8_t>> CompileMetalMatmulAirToMetallib() {
  return CompileMetalAirToMetallib(get_matmul_air_direct(),
                                   "metal_matmul_air");
}

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
  if (!IsScalarLikeF32(root->shape()) || reduce->operand_count() != 2 ||
      reduce->dimensions().size() != 1 || reduce->dimensions()[0] != 0) {
    return absl::UnimplementedError(
        "Metal direct AIR reduction currently supports only single f32 array "
        "to f32 scalar reductions over dimension 0.");
  }

  const HloInstruction* input = reduce->operand(0);
  const HloInstruction* init = reduce->operand(1);
  if (!IsF32Array(input->shape()) || input->shape().dimensions().size() != 1 ||
      !init->IsConstant() ||
      !IsScalarLikeF32(init->shape())) {
    return absl::UnimplementedError(
        "Metal direct AIR reduction currently supports only rank-1 f32 inputs "
        "and scalar f32 constant init values.");
  }

  int64_t input_start = 0;
  int64_t input_stride = 1;
  const HloInstruction* parameter = input;
  if (input->opcode() == HloOpcode::kSlice) {
    if (input->slice_strides().size() != 1 ||
        input->operand(0)->opcode() != HloOpcode::kParameter ||
        !IsF32Array(input->operand(0)->shape()) ||
        input->operand(0)->shape().dimensions().size() != 1) {
      return absl::UnimplementedError(
          "Metal direct AIR reduction currently supports only rank-1 f32 "
          "slices of parameters.");
    }
    input_start = input->slice_starts(0);
    input_stride = input->slice_strides(0);
    parameter = input->operand(0);
  }
  if (parameter->opcode() != HloOpcode::kParameter) {
    return absl::UnimplementedError(
        "Metal direct AIR reduction currently supports only parameter or slice "
        "inputs.");
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
    default:
      return absl::UnimplementedError(absl::StrFormat(
          "Metal direct AIR reduction does not support reducer opcode %s.",
          HloOpcodeString(reducer->opcode())));
  }

  MetalReductionConfig config;
  config.parameter_number = parameter->parameter_number();
  config.num_elements = ShapeUtil::ElementsIn(input->shape());
  config.input_start = input_start;
  config.input_stride = input_stride;
  config.init_value = init->literal().Get<float>({});
  config.output_scale = output_scale;
  config.kind = kind;
  return config;
}

std::string ReductionUpdate(ReductionKind kind) {
  switch (kind) {
    case ReductionKind::kAdd:
      return "  %new_acc = fadd fast float %acc, %value\n";
    case ReductionKind::kMultiply:
      return "  %new_acc = fmul fast float %acc, %value\n";
    case ReductionKind::kMaximum:
      return R"(  %cmp = fcmp fast ogt float %value, %acc
  %new_acc = select i1 %cmp, float %value, float %acc
)";
    case ReductionKind::kMinimum:
      return R"(  %cmp = fcmp fast olt float %value, %acc
  %new_acc = select i1 %cmp, float %value, float %acc
)";
  }
  return "";
}

std::string BuildReductionAir(const MetalReductionConfig& config) {
  return absl::StrFormat(R"(
source_filename = "xla/service/gpu/metal_reduction_air"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024-n8:16:32"
target triple = "air64_v27-apple-macosx15.0.0"

%%struct.ReductionParams = type { i32, i32, i32, i32 }

define void @reduce_f32(
    float addrspace(1)* nocapture noundef readonly "air-buffer-no-alias" %%arg0,
    float addrspace(1)* nocapture noundef writeonly "air-buffer-no-alias" %%out,
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
  %%acc = phi float [ %s, %%body ], [ %%new_acc, %%loop_body ]
  %%in_bounds = icmp ult i32 %%i, %%n
  br i1 %%in_bounds, label %%loop_body, label %%done

loop_body:
  %%i64 = zext i32 %%i to i64
  %%scaled_i = mul i64 %%i64, %d
  %%source_i = add i64 %%scaled_i, %d
  %%ptr = getelementptr inbounds float, float addrspace(1)* %%arg0, i64 %%source_i
  %%value = load float, float addrspace(1)* %%ptr, align 4
%s  %%next = add i32 %%i, 1
  br label %%loop

done:
  %%scaled_acc = fmul fast float %%acc, %s
  store float %%scaled_acc, float addrspace(1)* %%out, align 4
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

!0 = !{void (float addrspace(1)*, float addrspace(1)*, %%struct.ReductionParams addrspace(2)*, <3 x i32>)* @reduce_f32, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !7}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"arg0"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !6, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"ReductionParams", !"air.arg_name", !"params"}
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
!21 = !{!"xla direct AIR reduction"}
!22 = !{i32 2, i32 7, i32 0}
!23 = !{!"Metal", i32 3, i32 2, i32 0}
!24 = !{!"xla/service/gpu/metal_reduction_air"}
)",
                         FloatLiteral(config.init_value),
                         config.input_stride, config.input_start,
                         ReductionUpdate(config.kind),
                         FloatLiteral(config.output_scale));
}

class MetalReductionExecutable final : public Executable {
 public:
  MetalReductionExecutable(std::shared_ptr<HloModule> module,
                           MetalReductionConfig config,
                           std::vector<uint8_t> metallib)
      : Executable(std::move(module)),
        config_(config),
        result_shape_(this->module()
                          .entry_computation()
                          ->root_instruction()
                          ->shape()),
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
    ReductionParams params{static_cast<uint32_t>(config_.num_elements), 0, 0,
                           0};
    TF_RETURN_IF_ERROR(
        stream->Memcpy(&params_address, &params, sizeof(ReductionParams)));

    se::DeviceAddressBase input =
        arguments[config_.parameter_number].Buffer({}).AsDeviceAddress();
    if (input.is_null()) {
      return absl::InvalidArgumentError("Metal reduction input buffer is null.");
    }

    auto spec = se::KernelLoaderSpec::CreateOwningMetalLibraryInMemorySpec(
        std::vector<uint8_t>(metallib_.begin(), metallib_.end()), "reduce_f32",
        /*arity=*/3);
    TF_ASSIGN_OR_RETURN(std::unique_ptr<se::Kernel> kernel,
                        executor->LoadKernel(spec));

    se::KernelArgsPackedArray kernel_args(/*num_args=*/3);
    kernel_args.add_argument(input);
    kernel_args.add_argument(output);
    kernel_args.add_argument(params_address);

    TF_RETURN_IF_ERROR(kernel->Launch(se::ThreadDim(1, 1, 1),
                                      se::BlockDim(1, 1, 1), stream,
                                      kernel_args));
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
  TF_ASSIGN_OR_RETURN(std::vector<uint8_t> metallib,
                      CompileMetalAirToMetallib(BuildReductionAir(config),
                                                "metal_reduction_air"));
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
        result_shape_(this->module()
                          .entry_computation()
                          ->root_instruction()
                          ->shape()),
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
                             std::vector<uint8_t> metallib)
      : Executable(std::move(module)),
        parameter_numbers_(std::move(parameter_numbers)),
        result_shape_(this->module()
                          .entry_computation()
                          ->root_instruction()
                          ->shape()),
        num_elements_(ShapeUtil::ElementsIn(result_shape_)),
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

absl::StatusOr<std::unique_ptr<Executable>> BuildMetalElementwiseExecutable(
    std::shared_ptr<HloModule> module) {
  HloInstruction* root = module->entry_computation()->root_instruction();
  ElementwiseAirEmitter emitter(root->shape());
  TF_ASSIGN_OR_RETURN(std::string air, emitter.Emit(root));
  TF_ASSIGN_OR_RETURN(std::vector<uint8_t> metallib,
                      CompileMetalAirToMetallib(air, "metal_elementwise_air"));
  return std::make_unique<MetalElementwiseExecutable>(
      std::move(module), emitter.parameter_numbers(), std::move(metallib));
}

MetalMatmulExecutable::MetalMatmulExecutable(std::shared_ptr<HloModule> module,
                                             MetalMatmulConfig config,
                                             std::vector<uint8_t> metallib)
    : Executable(std::move(module)),
      config_(config),
      result_shape_(this->module()
                        .entry_computation()
                        ->root_instruction()
                        ->shape()),
      kernel_name_(config.relu ? "matmul_relu_simdgroup_8x8"
                               : "matmul_simdgroup_8x8"),
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
  se::BlockDim blocks(/*x=*/static_cast<uint64_t>((config_.n + 31) / 32),
                      /*y=*/static_cast<uint64_t>((config_.m + 15) / 16),
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
