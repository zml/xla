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
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/autotuner/block_level_emitter.h"
#include "xla/backends/gpu/autotuner/factory.h"
#include "xla/backends/gpu/autotuner/fission_backend.h"
#include "xla/backends/gpu/autotuner/gpu_codegen_backend.h"
#include "xla/backends/gpu/autotuner/native_emitter.h"
#include "xla/backends/gpu/autotuner/triton.h"
#include "xla/backends/gpu/transforms/dot_algorithm_rewriter.h"
#include "xla/backends/gpu/transforms/gemm_rewriter.h"
#include "xla/backends/gpu/transforms/scaled_dot_rewriter.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform/platform_object_registry.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

using ::mlir::MLIRContext;

bool AllowsBackend(absl::Span<const autotuner::Backend> backend_allowlist,
                   autotuner::Backend expected) {
  if (backend_allowlist.empty()) {
    return true;
  }
  for (autotuner::Backend backend : backend_allowlist) {
    if (backend == expected) {
      return true;
    }
  }
  return false;
}

std::unique_ptr<HloPassPipeline> GetOneMklRewriterPipeline(
    const stream_executor::DeviceDescription& device_description) {
  auto pipeline = std::make_unique<HloPassPipeline>("onemkl_rewriter_pipeline");
  pipeline->AddPass(std::make_unique<DotAlgorithmRewriter>());
  pipeline->AddPass(std::make_unique<ScaledDotRewriter>());
  for (GemmRewriterOptions::DType dtype :
       {GemmRewriterOptions::DType::kFp8Only,
        GemmRewriterOptions::DType::kNonFp8Only}) {
    GemmRewriterOptions options{dtype};
    options.enable_cublaslt = false;
    pipeline->AddPass(std::make_unique<GemmRewriter>(
        device_description.gpu_compute_capability(),
        device_description.runtime_version(), options));
  }
  return pipeline;
}

class OneMklBackend : public GpuCodegenBackend {
 public:
  explicit OneMklBackend(stream_executor::StreamExecutor* stream_executor,
                         const DebugOptions* debug_options, Compiler* compiler,
                         const Compiler::GpuTargetConfig* target_config)
      : GpuCodegenBackend(autotuner::Backend::ONEMKL, debug_options, compiler,
                          target_config, stream_executor,
                          /*uses_last_output_for_scratch=*/true) {}

  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
  GetSupportedConfigs(const HloInstruction& instr) override {
    std::vector<std::unique_ptr<BackendConfig>> configs;
    if (!IsSupported(instr)) {
      return configs;
    }
    TF_ASSIGN_OR_RETURN(std::unique_ptr<BackendConfig> config,
                        GetDefaultConfig(instr));
    configs.push_back(std::move(config));
    return configs;
  }

  absl::StatusOr<std::unique_ptr<BackendConfig>> GetDefaultConfig(
      const HloInstruction& instr) override {
    if (!IsSupported(instr)) {
      return absl::InvalidArgumentError(
          "OneMklBackend does not support this instruction.");
    }
    AutotuneResult::GemmKey gemm_key;
    gemm_key.set_algorithm(stream_executor::blas::kDefaultAlgorithm);
    auto any = std::make_unique<google::protobuf::Any>();
    any->PackFrom(gemm_key);
    return any;
  }

  absl::Status ApplyConfig(HloInstruction& instr,
                           const BackendConfig& config) override {
    AutotuneResult::GemmKey gemm_key;
    if (!config.UnpackTo(&gemm_key)) {
      return absl::InvalidArgumentError(
          "Failed to unpack OneMklBackend GemmKey from Any.");
    }
    TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                        instr.backend_config<GpuBackendConfig>());
    GemmBackendConfig& backend_config =
        *gpu_config.mutable_gemm_backend_config();
    backend_config.set_selected_algorithm(gemm_key.algorithm());
    backend_config.set_autotune_workspace_size(
        gemm_key.autotune_workspace_size());
    return instr.set_backend_config(std::move(gpu_config));
  }

 private:
  bool IsSupported(const HloInstruction& instr) override {
    return IsLegacyCublasMatmul(instr);
  }
};

}  // namespace

std::vector<std::unique_ptr<CodegenBackend>> GetCodegenBackendsForSycl(
    stream_executor::StreamExecutor* stream_executor,
    stream_executor::DeviceAddressAllocator*, const DebugOptions* debug_options,
    Compiler* compiler, const Compiler::GpuTargetConfig* target_config,
    const AliasInfo* alias_info, MLIRContext* mlir_context,
    HloCostAnalysis::ShapeSizeFunction shape_size_fn,
    absl::Span<const autotuner::Backend> backend_allowlist) {
  std::vector<std::unique_ptr<CodegenBackend>> backends;
  if (AllowsBackend(backend_allowlist, autotuner::Backend::TRITON)) {
    backends.push_back(std::make_unique<TritonBackend>(
        debug_options, compiler, target_config, alias_info, mlir_context));
  }
  if (AllowsBackend(backend_allowlist, autotuner::Backend::ONEMKL)) {
    backends.push_back(std::make_unique<OneMklBackend>(
        stream_executor, debug_options, compiler, target_config));
  }
  if (AllowsBackend(backend_allowlist, autotuner::Backend::ONEMKL_FISSION)) {
    backends.push_back(std::make_unique<FissionBackend>(
        debug_options, compiler, target_config,
        std::make_unique<OneMklBackend>(stream_executor, debug_options,
                                        compiler, target_config),
        GetOneMklRewriterPipeline(target_config->device_description),
        alias_info, mlir_context, stream_executor));
  }
  if (AllowsBackend(backend_allowlist, autotuner::Backend::NATIVE_EMITTER)) {
    backends.push_back(std::make_unique<NativeEmitterBackend>(
        debug_options, compiler, target_config));
  }
  if (AllowsBackend(backend_allowlist,
                    autotuner::Backend::BLOCK_LEVEL_EMITTER)) {
    backends.push_back(std::make_unique<BlockLevelEmitterBackend>(
        debug_options, compiler, shape_size_fn, target_config));
  }
  return backends;
}

STREAM_EXECUTOR_REGISTER_OBJECT_STATICALLY(GetCodegenBackendsSyclRegistration,
                                           GetCodegenBackends,
                                           se::sycl::kSyclPlatformId,
                                           GetCodegenBackendsForSycl);

}  // namespace gpu
}  // namespace xla
