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
#include "xla/service/gpu/gpu_compiler.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

// MetalGpuCompiler reparents the Apple-Metal backend onto xla::gpu::GpuCompiler
// so it reuses the full HLO pipeline + buffer assignment + fusion + thunk
// runtime, and emits Apple AIR (textual .ll -> air-as -> metallib) at the
// CompileTargetBinary seam instead of entering LLVM's NVPTX/AMDGPU backend.
//
// This is the sole Metal compiler. See metal-xla-docs/GPUCOMPILER_PORT_PLAN.md
// for the design; the overrides below prune the CUDA/AMD-centric passes the base
// GpuCompiler runs and wire up the Metal fusion/gemm codegen seams.
class MetalGpuCompiler : public GpuCompiler {
 public:
  MetalGpuCompiler();

  // Force single-device before the standard GpuCompiler HLO pipeline. ZML
  // configures num_partitions>1 (+ sdy.sharding) for its replicated mesh, which
  // makes GpuCompiler::RunSPMDPasses run the real SPMD partitioner — that packs
  // entry parameters into a partitioned temp allocation, so on one Metal device
  // multi-input fusions read their inputs as zero. With
  // num_partitions==1 the gate `num_partitions>1 && use_spmd_partitioning` is
  // false, so the pipeline instead runs the sharding-REMOVAL branch
  // (ShardingRemover + ShardyXLA(no-propagation) + HloDCE), stripping the
  // Shardy/sharding wrappers and running unpartitioned — matching the legacy
  // backend's single-device assumption. Then delegate to the base pipeline.
  absl::StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;

  // After scaled-dot lowering and layout assignment, make active prefix counts
  // and mutable Metal custom-call workspaces explicit before fusion and
  // mandatory copy insertion. This keeps BufferAssignment responsible for
  // both dependency and scratch lifetime.
  absl::Status OptimizeHloPostLayoutAssignment(
      HloModule* hlo_module, se::StreamExecutor* stream_exec,
      const CompileOptions& options,
      const GpuTargetConfig& gpu_target_config,
      const GpuAliasInfo* alias_info, tsl::thread::ThreadPool* thread_pool,
      CompilationStats* compilation_stats,
      mlir::MLIRContext* mlir_context) override;

  // No cuDNN/MIOpen convolution rewriting on Metal.
  absl::Status OptimizeHloConvolutionCanonicalization(
      HloModule* hlo_module, const se::GpuComputeCapability& gpu_version,
      se::dnn::VersionInfo dnn_version,
      const se::SemanticVersion& toolkit_version,
      CompilationStats* compilation_stats) override;

  // No cuBLAS gemm padding on Metal (metalBLAS handles any shape). Reachable
  // because EnableGemmCustomCalls() stays true so dots become __cublas$gemm.
  void AddPaddingForGpublasGemms(
      HloPassPipeline& pipeline, const DebugOptions& debug_options,
      const se::GpuComputeCapability& gpu_version) override;

  // THE codegen seam: take the (AIR-targeted) llvm::Module and assemble it to a
  // .metallib via the shared air-as toolchain. Never enters NVPTX/AMDGPU.
  absl::StatusOr<BackendCompileResult> CompileTargetBinary(
      const HloModuleConfig& module_config, llvm::Module* llvm_module,
      const se::DeviceDescription& device_description, bool relocatable,
      const HloModule* debug_module, std::optional<int> shard_number) override;

  // Metal has no DnnSupport; without this RunBackend TF_RET_CHECKs.
  bool RequiresDnnSupport() const override { return false; }

  // Single deterministic AIR lowering; no autotuning of candidate kernels.
  bool EnableFusionAutotuning() const override { return false; }

  // No conv/gemm autotuning on Metal — add NO pass. The base impl calls
  // GetAutotunerBackends -> PlatformObjectRegistry::FindObject<GetCodegenBackends>
  // (PlatformId()), which has no Metal registration and fails NotFound at
  // compile time. Metal gemm goes to metalBLAS (no autotuning).
  absl::Status AddAutotunerPass(
      HloPassPipeline* pipeline, HloModule* hlo_module,
      const se::GpuComputeCapability& gpu_version,
      const CompileOptions& options, tsl::thread::ThreadPool* thread_pool,
      stream_executor::StreamExecutor* stream_executor,
      const GpuTargetConfig* target_config, const AliasInfo* alias_info,
      mlir::MLIRContext* mlir_context,
      HloCostAnalysis::ShapeSizeFunction shape_size_fn,
      const MultiProcessKeyValueStore& key_value_store) override;

  // NOTE: EnableGemmCustomCalls() intentionally left at the base default (true)
  // so dots become __cublas$gemm custom-calls and route to the Metal gemm thunk
  // (metalBLAS); if false they would be FusionWrapper-wrapped into kLoop fusions
  // our elementwise emitter cannot lower.
  //
  // TODO: prune the CUDA-centric portion of OptimizeHloPostLayoutAssignment
  // (cublas-LT / Triton / cuDNN-normalization) passes the base body runs
  // ungated, keeping only the layout/float/reduction normalization the
  // row-major AIR emitter needs. The override currently delegates to the base
  // body before appending the Metal workspace pass.

 private:
  MetalGpuCompiler(const MetalGpuCompiler&) = delete;
  MetalGpuCompiler& operator=(const MetalGpuCompiler&) = delete;

  // We never enter LLVM's target backend, so no target-specific LLVM flags.
  std::vector<std::string> GetLLVMCommandLineOptions(
      const DebugOptions& debug_options) const override;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_METAL_GPU_COMPILER_H_
