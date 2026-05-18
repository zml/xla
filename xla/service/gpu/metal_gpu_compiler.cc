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

#include "xla/service/gpu/metal_gpu_compiler.h"

#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/IR/Module.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/gpu_compiler.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/metal/metal_platform_id.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

namespace {

constexpr char kMetalTargetTriple[] = "air64-apple-macosx";
constexpr char kMetalDataLayout[] = "e-p:64:64-i64:64-n32:64-S128";

}  // namespace

MetalGpuCompiler::MetalGpuCompiler()
    : GpuCompiler(stream_executor::metal::kMetalPlatformId, kMetalTargetTriple,
                  kMetalDataLayout) {}

absl::Status MetalGpuCompiler::OptimizeHloConvolutionCanonicalization(
    HloModule* hlo_module, const se::GpuComputeCapability& gpu_version,
    se::dnn::VersionInfo dnn_version,
    const se::SemanticVersion& toolkit_version,
    CompilationStats* compilation_stats) {
  return absl::OkStatus();
}

void MetalGpuCompiler::AddPaddingForGpublasGemms(
    HloPassPipeline& pipeline, const DebugOptions& debug_options,
    const se::GpuComputeCapability& gpu_version) {}

absl::Status MetalGpuCompiler::AddConvAndGemmAutotuningPass(
    HloPassPipeline* pipeline, HloModule* hlo_module,
    const se::GpuComputeCapability& gpu_version, const CompileOptions& options,
    tsl::thread::ThreadPool* thread_pool, se::StreamExecutor* stream_exec,
    const Compiler::GpuTargetConfig* target_config,
    const MultiProcessKeyValueStore& key_value_store,
    const se::SemanticVersion& toolkit_version, const AliasInfo* alias_info,
    const DebugOptions& debug_options, mlir::MLIRContext* mlir_context,
    HloCostAnalysis::ShapeSizeFunction shape_size_fn) {
  return absl::OkStatus();
}

absl::StatusOr<GpuCompiler::BackendCompileResult>
MetalGpuCompiler::CompileTargetBinary(
    const HloModuleConfig& module_config, llvm::Module* llvm_module,
    const stream_executor::DeviceDescription& device_description,
    bool relocatable, const HloModule* debug_module,
    std::optional<int> shard_number) {
  return absl::UnimplementedError(
      "XLA:GPU Metal requires a direct MLIR-to-MSL emitter. The backend is "
      "registered, but direct MSL code generation is not implemented yet.");
}

std::vector<std::string> MetalGpuCompiler::GetLLVMCommandLineOptions(
    const DebugOptions& debug_options) const {
  return {};
}

}  // namespace gpu
}  // namespace xla
