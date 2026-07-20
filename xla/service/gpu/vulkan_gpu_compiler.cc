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

#include "xla/service/gpu/vulkan_gpu_compiler.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "xla/service/dump.h"
#include "xla/service/gpu/llvm_gpu_backend/spirv_backend.h"
#include "xla/service/gpu/target_constants.h"
#include "xla/stream_executor/vulkan/vulkan_platform_id.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla::gpu {

VulkanGpuCompiler::VulkanGpuCompiler()
    : GpuCompiler(stream_executor::vulkan::kVulkanPlatformId,
                  vulkan::TargetTriple(), vulkan::DataLayout()) {}

absl::Status VulkanGpuCompiler::OptimizeHloConvolutionCanonicalization(
    HloModule*, const se::GpuComputeCapability&, se::dnn::VersionInfo,
    const se::SemanticVersion&, CompilationStats*) {
  return absl::OkStatus();
}

void VulkanGpuCompiler::AddPaddingForGpublasGemms(
    HloPassPipeline&, const DebugOptions&,
    const se::GpuComputeCapability&) {}

absl::Status VulkanGpuCompiler::AddAutotunerPass(
    HloPassPipeline*, HloModule*, const se::GpuComputeCapability&,
    const CompileOptions&, tsl::thread::ThreadPool*,
    stream_executor::StreamExecutor*, const GpuTargetConfig*,
    const AliasInfo*, mlir::MLIRContext*,
    HloCostAnalysis::ShapeSizeFunction, const MultiProcessKeyValueStore&) {
  return absl::OkStatus();
}

absl::StatusOr<GpuCompiler::BackendCompileResult>
VulkanGpuCompiler::CompileTargetBinary(
    const HloModuleConfig& module_config, llvm::Module* llvm_module,
    const stream_executor::DeviceDescription& device_description,
    bool relocatable, const HloModule* debug_module,
    std::optional<int> shard_number) {
  if (relocatable) {
    return Unimplemented("Relocatable Vulkan SPIR-V is not supported.");
  }

  bool has_entry_point = false;
  for (const llvm::Function& function : *llvm_module) {
    has_entry_point |=
        !function.isDeclaration() && function.hasFnAttribute("hlsl.shader");
  }
  if (!has_entry_point) {
    return BackendCompileResult{/*asm_text=*/"", /*binary=*/{}};
  }

  ASSIGN_OR_RETURN(
      std::string spirv,
      spirv::CompileToVulkanSPIRV(
          llvm_module, device_description.gpu_compute_capability(),
          module_config.debug_options()));
  if (DumpingEnabledForHloModule(debug_module ? debug_module->name() : "",
                                 module_config.debug_options()) &&
      debug_module != nullptr) {
    DumpToFileInDirOrStdout(
        *debug_module, "",
        shard_number.has_value()
            ? (std::to_string(*shard_number) + ".vulkan.spv")
            : "vulkan.spv",
        spirv);
  }

  return BackendCompileResult{
      /*asm_text=*/"",
      /*binary=*/std::vector<uint8_t>(spirv.begin(), spirv.end())};
}

std::vector<std::string> VulkanGpuCompiler::GetLLVMCommandLineOptions(
    const DebugOptions& debug_options) const {
  return spirv::GetSPIRVBackendOptions(debug_options);
}

}  // namespace xla::gpu
