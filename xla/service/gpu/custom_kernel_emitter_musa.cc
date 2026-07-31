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

#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/codegen/kernel_compiler.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/future.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/custom_kernel_emitter.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/kernel_call.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/musa/musa_custom_kernel_source.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla::gpu {
namespace {

absl::Status ValidateMusaCustomKernelLaunch(
    const KernelCall& call, const se::DeviceDescription& device) {
  if (!se::ThreadDimOk(device, call.thread_dim)) {
    return absl::InvalidArgumentError(
        "MUSA custom kernel thread dimensions exceed device limits");
  }
  const se::BlockDim& limit = device.block_dim_limit();
  if (call.block_dim.x > limit.x || call.block_dim.y > limit.y ||
      call.block_dim.z > limit.z) {
    return absl::InvalidArgumentError(
        "MUSA custom kernel grid dimensions exceed device limits");
  }
  if (call.shared_mem != 0) {
    return absl::UnimplementedError(
        "MUSA native custom kernels do not yet support dynamic shared memory");
  }
  return absl::OkStatus();
}

}  // namespace

xla::Future<std::unique_ptr<Thunk>> EmitCustomKernelThunk(
    const HloCustomCallInstruction* instr, IrEmitterContext* context) {
  if (!context->gpu_compute_capability().IsMusa()) {
    return absl::InvalidArgumentError(
        "MUSA custom kernel emitter requires a MUSA compute capability");
  }
  if (instr->custom_call_target() != kMusaLlvmCustomCallTarget) {
    return absl::UnimplementedError(
        "MUSA does not accept PTX or CUDA binary custom kernels; use the "
        "versioned MUSA LLVM custom-call target");
  }

  absl::string_view backend_config = instr->raw_backend_config_string();
  if (backend_config.empty()) {
    return absl::InvalidArgumentError(
        "MUSA LLVM custom call backend config is empty");
  }
  ASSIGN_OR_RETURN(
      KernelCall call,
      KernelCall::Parse(backend_config, context->mlir_context()));
  if (call.kernel_type != KernelCall::KernelType::kMusaLlvmSource) {
    return absl::InvalidArgumentError(
        "MUSA LLVM custom call backend config is not current LLVM source");
  }
  RETURN_IF_ERROR(
      ValidateMusaCustomKernelLaunch(call, context->gpu_device_info()));

  ASSIGN_OR_RETURN(
      LlvmKernelSource source,
      musa::ParseMusaCustomKernelSource(call.name, call.kernel_data));
  ASSIGN_OR_RETURN(
      emitters::KernelArguments kernel_arguments,
      emitters::KernelArguments::Create(
          context->buffer_assignment(), GetDefaultBufferAlignment(), instr,
          call.output_indices));

  Thunk::ThunkInfo thunk_info =
      Thunk::ThunkInfo::WithProfileAnnotation(instr, context->GetNextThunkId());
  return context->kernel_compiler()->Compile(
      std::move(thunk_info), std::move(source), call.name, kernel_arguments,
      LaunchDimensions(call.block_dim, call.thread_dim));
}

}  // namespace xla::gpu
