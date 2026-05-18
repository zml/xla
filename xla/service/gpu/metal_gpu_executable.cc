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
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
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

bool IsF32Rank2Array(const Shape& shape) {
  return shape.element_type() == F32 && shape.dimensions().size() == 2;
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
  TF_ASSIGN_OR_RETURN(std::string air_as, FindMetalTool("AIR_AS", "air-as"));
  TF_ASSIGN_OR_RETURN(std::string air_opt,
                      FindMetalTool("AIR_OPT", "air-opt"));
  TF_ASSIGN_OR_RETURN(std::string metallib,
                      FindMetalTool("METALLIB", "metallib"));

  tsl::Env* env = tsl::Env::Default();
  std::string base;
  if (!env->LocalTempFilename(&base)) {
    return absl::InternalError("Could not create Metal AIR temp filename.");
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

  TF_RETURN_IF_ERROR(
      tsl::WriteStringToFile(env, air_ll_path, get_matmul_air_direct()));
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
