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

#ifndef XLA_SERVICE_GPU_METAL_GPU_COMPILER_H_
#define XLA_SERVICE_GPU_METAL_GPU_COMPILER_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/IR/Module.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/compiler.h"
#include "xla/backends/gpu/transforms/fused_scaled_dot_arms_metal.h"
#include "xla/service/gpu/gpu_compiler.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

class MetalGpuCompiler : public GpuCompiler {
 public:
  MetalGpuCompiler();

  absl::StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;

  absl::Status OptimizeHloPostLayoutAssignment(
      HloModule* hlo_module, se::StreamExecutor* stream_exec,
      const CompileOptions& options,
      const GpuTargetConfig& gpu_target_config,
      const GpuAliasInfo* alias_info, tsl::thread::ThreadPool* thread_pool,
      CompilationStats* compilation_stats,
      mlir::MLIRContext* mlir_context) override;

  absl::Status OptimizeHloConvolutionCanonicalization(
      HloModule* hlo_module, const se::GpuComputeCapability& gpu_version,
      se::dnn::VersionInfo dnn_version,
      const se::SemanticVersion& toolkit_version,
      CompilationStats* compilation_stats) override;

  void AddPaddingForGpublasGemms(
      HloPassPipeline& pipeline, const DebugOptions& debug_options,
      const se::GpuComputeCapability& gpu_version) override;

  absl::StatusOr<BackendCompileResult> CompileTargetBinary(
      const HloModuleConfig& module_config, llvm::Module* llvm_module,
      const se::DeviceDescription& device_description, bool relocatable,
      const HloModule* debug_module, std::optional<int> shard_number) override;

  bool RequiresDnnSupport() const override { return false; }

  std::vector<FusedScaledDotArm> FusedScaledDotArms(
      FusedScaledDotPhase phase, const DebugOptions& debug_options,
      const GpuTargetConfig& gpu_target_config) const override {
    return MetalFusedScaledDotArms();
  }

  bool EnableFusionAutotuning() const override { return false; }

  absl::Status AddAutotunerPass(
      HloPassPipeline* pipeline, HloModule* hlo_module,
      const se::GpuComputeCapability& gpu_version,
      const CompileOptions& options, tsl::thread::ThreadPool* thread_pool,
      stream_executor::StreamExecutor* stream_executor,
      const GpuTargetConfig* target_config, const AliasInfo* alias_info,
      mlir::MLIRContext* mlir_context,
      HloCostAnalysis::ShapeSizeFunction shape_size_fn,
      const MultiProcessKeyValueStore& key_value_store) override;

 private:
  MetalGpuCompiler(const MetalGpuCompiler&) = delete;
  MetalGpuCompiler& operator=(const MetalGpuCompiler&) = delete;

  std::vector<std::string> GetLLVMCommandLineOptions(
      const DebugOptions& debug_options) const override;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_METAL_GPU_COMPILER_H_
