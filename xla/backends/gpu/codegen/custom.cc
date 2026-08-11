/* Copyright 2024 The OpenXLA Authors.

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
#include "xla/backends/gpu/codegen/custom.h"

#include <memory>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/backends/gpu/codegen/emitters/vulkan_flash_attention.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

AsyncThunkSequence CustomFusion::Emit(
    IrEmitterContext& ir_emitter_context,
    const HloFusionInstruction& fusion) const {
  std::optional<std::string> config_name =
      GetCustomFusionConfigName(&fusion);
  if (config_name.has_value() &&
      absl::string_view(*config_name) ==
          kVulkanFlashAttentionFusionConfigName) {
    const se::DeviceDescription& device =
        ir_emitter_context.gpu_device_info();
    const se::VulkanComputeCapability* capability =
        device.gpu_compute_capability().vulkan_compute_capability();
    if (capability == nullptr || !capability->shader_bfloat16() ||
        !capability->storage_buffer_16bit_access() ||
        !capability->subgroup_basic() || !capability->subgroup_shuffle()) {
      return absl::FailedPreconditionError(
          "Vulkan flash-attention requires BF16 arithmetic, 16-bit storage, "
          "and subgroup basic and shuffle operations");
    }
    MlirKernelFusion emitter(std::make_unique<VulkanFlashAttentionEmitter>(
        fusion, ir_emitter_context.gpu_device_info()));
    return emitter.Emit(ir_emitter_context, fusion);
  }
  return absl::UnimplementedError("Custom kernel fusion is not supported.");
}

}  // namespace gpu
}  // namespace xla
