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

#include "xla/service/gpu/musa_gpu_compiler.h"

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
#include "xla/service/executable.h"
#include "xla/service/gpu/target_constants.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/musa/musa_platform_id.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla::gpu {
namespace {

absl::Status MusaCompilationUnimplemented() {
  return absl::UnimplementedError(
      "MUSA XLA compilation is not implemented in PJRT GPU v1. "
      "Build and platform plumbing are available; compiled HLO execution ");
}

}  // namespace

MusaGpuCompiler::MusaGpuCompiler()
    : GpuCompiler(stream_executor::musa::kMusaPlatformId, musa::TargetTriple(),
                  musa::DataLayout()) {}

se::Platform::Id MusaGpuCompiler::PlatformId() const {
  return stream_executor::musa::kMusaPlatformId;
}

absl::StatusOr<std::unique_ptr<HloModule>> MusaGpuCompiler::RunHloPasses(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
    const CompileOptions& options) {
  (void)module;
  (void)stream_exec;
  (void)options;
  return MusaCompilationUnimplemented();
}

absl::StatusOr<std::unique_ptr<Executable>> MusaGpuCompiler::RunBackend(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
    const CompileOptions& options) {
  (void)module;
  (void)stream_exec;
  (void)options;
  return MusaCompilationUnimplemented();
}

absl::StatusOr<std::vector<std::unique_ptr<CompiledModule>>>
MusaGpuCompiler::CompileAheadOfTime(std::unique_ptr<HloModule> hlo_module,
                                    const AotCompilationOptions& options) {
  (void)hlo_module;
  (void)options;
  return MusaCompilationUnimplemented();
}

std::vector<std::string> MusaGpuCompiler::GetLLVMCommandLineOptions(
    const DebugOptions& debug_options) const {
  (void)debug_options;
  return {};
}

void MusaGpuCompiler::AddPaddingForGpublasGemms(
    HloPassPipeline& pipeline, const DebugOptions& debug_options,
    const se::GpuComputeCapability& gpu_version) {
  (void)pipeline;
  (void)debug_options;
  (void)gpu_version;
}

absl::Status MusaGpuCompiler::OptimizeHloConvolutionCanonicalization(
    HloModule* hlo_module, const se::GpuComputeCapability& gpu_version,
    se::dnn::VersionInfo dnn_version,
    const se::SemanticVersion& toolkit_version,
    CompilationStats* compilation_stats) {
  (void)hlo_module;
  (void)gpu_version;
  (void)dnn_version;
  (void)toolkit_version;
  (void)compilation_stats;
  return MusaCompilationUnimplemented();
}

absl::StatusOr<GpuCompiler::BackendCompileResult>
MusaGpuCompiler::CompileTargetBinary(
    const HloModuleConfig& module_config, llvm::Module* llvm_module,
    const stream_executor::DeviceDescription& device_description,
    bool relocatable, const HloModule* debug_module,
    std::optional<int> shard_number) {
  (void)module_config;
  (void)llvm_module;
  (void)device_description;
  (void)relocatable;
  (void)debug_module;
  (void)shard_number;
  return MusaCompilationUnimplemented();
}

absl::StatusOr<bool> MusaGpuCompiler::CanUseLinkModules(
    const HloModuleConfig& module_config,
    const stream_executor::DeviceDescription& device_description,
    se::StreamExecutor* absl_nullable stream_exec) {
  (void)module_config;
  (void)device_description;
  (void)stream_exec;
  return false;
}

}  // namespace xla::gpu
