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

#ifndef XLA_SERVICE_GPU_MUSA_GPU_COMPILER_H_
#define XLA_SERVICE_GPU_MUSA_GPU_COMPILER_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/IR/Module.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/service/compilation_stats.h"
#include "xla/service/compiled_module.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_compiler.h"
#include "xla/service/gpu/musa/musa_compilation_provider.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/abi/executable_abi_version.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla::gpu {

class MusaGpuCompiler : public GpuCompiler {
 public:
  MusaGpuCompiler();
  explicit MusaGpuCompiler(
      std::unique_ptr<musa::MusaCompilationProvider> compilation_provider);

  se::Platform::Id PlatformId() const override;

  std::vector<std::string> GetLLVMCommandLineOptions(
      const DebugOptions& debug_options) const override;

  void AddPaddingForGpublasGemms(
      HloPassPipeline& pipeline, const DebugOptions& debug_options,
      const se::GpuComputeCapability& gpu_version) override;

  void AddGemmRewriterPasses(
      HloPassPipeline& pipeline, const DebugOptions& debug_options,
      const se::GpuComputeCapability& gpu_version,
      const se::SemanticVersion& toolkit_version) override;

  absl::Status OptimizeHloConvolutionCanonicalization(
      HloModule* hlo_module, const se::GpuComputeCapability& gpu_version,
      se::dnn::VersionInfo dnn_version,
      const se::SemanticVersion& toolkit_version, bool is_deviceless,
      CompilationStats* compilation_stats) override;

  absl::Status OptimizeHloPostLayoutAssignment(
      HloModule* hlo_module, se::StreamExecutor* stream_exec,
      const CompileOptions& options, const GpuTargetConfig& gpu_target_config,
      const GpuAliasInfo* alias_info, tsl::thread::ThreadPool* thread_pool,
      CompilationStats* compilation_stats,
      mlir::MLIRContext* mlir_context) override;

  absl::StatusOr<BackendCompileResult> CompileTargetBinary(
      const HloModuleConfig& module_config, llvm::Module* llvm_module,
      const stream_executor::DeviceDescription& device_description,
      bool relocatable, const HloModule* debug_module,
      std::optional<int> shard_number) override;

  absl::StatusOr<bool> CanUseLinkModules(
      const HloModuleConfig& module_config,
      const stream_executor::DeviceDescription& device_description,
      se::StreamExecutor* absl_nullable stream_exec) override;

 protected:
  absl::StatusOr<stream_executor::ExecutableAbiVersion>
  CreateExecutableAbiVersion(
      const HloModule& module,
      const stream_executor::DeviceDescription& device_description,
      absl::Span<const uint8_t> main_binary) const override;

  bool UseAotCompiledThunks(const HloModule& module) const override;

  absl::Status ValidatePersistentKernelCache(
      const HloModuleConfig& module_config) const override;

  absl::Status AddAutotunerPass(
      HloPassPipeline* pipeline, HloModule* hlo_module,
      const se::GpuComputeCapability& gpu_version,
      const CompileOptions& options, tsl::thread::ThreadPool* thread_pool,
      se::StreamExecutor* stream_executor, const GpuTargetConfig* target_config,
      const AliasInfo* alias_info, mlir::MLIRContext* mlir_context,
      HloCostAnalysis::ShapeSizeFunction shape_size_fn,
      const MultiProcessKeyValueStore& key_value_store) override;

 private:
  std::unique_ptr<musa::MusaCompilationProvider> compilation_provider_;
  absl::Status compilation_provider_status_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_MUSA_GPU_COMPILER_H_
