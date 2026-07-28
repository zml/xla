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

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "llvm/IR/Module.h"
#include "xla/backends/gpu/transforms/dot_algorithm_rewriter.h"
#include "xla/backends/gpu/transforms/gemm_rewriter.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/service/compilation_stats.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/musa/musa_compilation_provider.h"
#include "xla/service/gpu/musa/musa_compiler_bundle.h"
#include "xla/service/gpu/musa/musa_executable_envelope.h"
#include "xla/service/gpu/musa/musa_llvm14_compatibility.h"
#include "xla/service/gpu/musa/protocol.h"
#include "xla/service/gpu/target_constants.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/musa/musa_mublas_api.h"
#include "xla/stream_executor/musa/musa_mufft_api.h"
#include "xla/stream_executor/musa/musa_optional_library_abi.h"
#include "xla/stream_executor/musa/musa_platform_id.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla::gpu {

namespace {

absl::Status ProviderUnavailable(const absl::Status& status) {
  return absl::Status(status.code(),
                      absl::StrCat("MUSA compilation provider is unavailable: ",
                                   status.message()));
}

std::vector<musa::MusaOptionalLibraryAbi> RequiredMusaOptionalLibraries(
    const HloModule& module) {
  bool requires_mublas = false;
  bool requires_mufft = false;
  bool requires_mublas_scal = false;
  // Deterministic execution configures the muBLAS atomics mode before every
  // GEMM. That setter is part of the v2 table even when the HLO itself is an
  // unbatched GEMM using the default algorithm.
  bool requires_advanced_mublas =
      module.config().debug_options().xla_gpu_deterministic_ops() ||
      module.config().debug_options().xla_gpu_exclude_nondeterministic_ops();
  for (const HloComputation* computation : module.computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kFft) {
        requires_mufft = true;
        requires_mublas_scal |= instruction->fft_type() == FftType::IFFT ||
                                instruction->fft_type() == FftType::IRFFT;
      }
      if (!IsMusaGemm(*instruction)) continue;
      requires_mublas = true;

      // C15 executables carried no fingerprint and remain loadable with either
      // the v1 or v2 shim table. A selected algorithm or a batched call depends
      // on the C16 v2 contract, so serialize its exact normalized capability
      // fingerprint and reject stale or downgraded runtime shims at load time.
      absl::StatusOr<GpuBackendConfig> gpu_config =
          instruction->backend_config<GpuBackendConfig>();
      if (!gpu_config.ok()) continue;
      const GemmBackendConfig& gemm = gpu_config->gemm_backend_config();
      const DotDimensionNumbers& dimensions = gemm.dot_dimension_numbers();
      requires_advanced_mublas |= gemm.has_selected_algorithm() ||
                                  dimensions.lhs_batch_dimensions_size() != 0 ||
                                  dimensions.rhs_batch_dimensions_size() != 0;
    }
  }
  std::vector<musa::MusaOptionalLibraryAbi> requirements;
  if (requires_mublas) {
    requirements.push_back({
        .name = stream_executor::musa::kMusaMuBlasLibraryAbiName,
        .abi_version = stream_executor::musa::kMusaMuBlasLibraryAbiVersion,
        .fingerprint =
            requires_advanced_mublas
                ? stream_executor::musa::kMusaMuBlasAdvancedAbiFingerprintV2
                : "",
    });
  }
  if (requires_mublas_scal) {
    requirements.push_back({
        .name = stream_executor::musa::kMusaMuBlasScalLibraryAbiName,
        .abi_version = stream_executor::musa::kMusaMuBlasScalLibraryAbiVersion,
        .fingerprint = stream_executor::musa::kMusaMuBlasScalAbiFingerprintV1,
    });
  }
  if (requires_mufft) {
    requirements.push_back({
        .name = stream_executor::musa::kMusaMuFftLibraryAbiName,
        .abi_version = stream_executor::musa::kMusaMuFftLibraryAbiVersion,
        .fingerprint = stream_executor::musa::kMusaMuFftAbiFingerprintV1,
    });
  }
  std::sort(requirements.begin(), requirements.end(),
            [](const musa::MusaOptionalLibraryAbi& lhs,
               const musa::MusaOptionalLibraryAbi& rhs) {
              return lhs.name < rhs.name;
            });
  return requirements;
}

absl::StatusOr<musa::MusaCompilationOptions> GetMusaCompilationOptions(
    const HloModuleConfig& module_config) {
  const DebugOptions& debug_options = module_config.debug_options();
  if (debug_options.xla_backend_optimization_level() < 2 ||
      debug_options.xla_backend_optimization_level() > 3) {
    return absl::UnimplementedError(
        "MUSA compilation currently supports XLA backend optimization "
        "levels 2 and 3 through the qualified vendor O2 profile");
  }
  if (debug_options.xla_enable_fast_math()) {
    return absl::UnimplementedError(
        "MUSA compilation does not yet support xla_enable_fast_math");
  }
  // The v1 provider contract is deliberately narrower than XLA's generic
  // debug options: deterministic vendor O2 with numerical controls disabled.
  return musa::MusaCompilationOptions();
}

std::string SafeMusaModuleName(absl::string_view name) {
  std::string result;
  result.reserve(std::min(name.size(), musa::kMusaBridgeMaxModuleNameBytes));
  for (char c : name) {
    const bool first = result.empty();
    const bool safe = absl::ascii_isalnum(c) || c == '_' ||
                      (!first && (c == '-' || c == '.' || c == '+'));
    if (first && !safe) result = "m_";
    result.push_back(safe ? c : '_');
    if (result.size() == musa::kMusaBridgeMaxModuleNameBytes) break;
  }
  return result.empty() ? "xla_musa_module" : result;
}

}  // namespace

MusaGpuCompiler::MusaGpuCompiler()
    : GpuCompiler(stream_executor::musa::kMusaPlatformId, musa::TargetTriple(),
                  musa::DataLayout()),
      compilation_provider_status_(absl::OkStatus()) {
  absl::StatusOr<std::unique_ptr<musa::MusaCompilationProvider>> provider =
      musa::LoadMusaCompilationProviderFromBundle();
  if (!provider.ok()) {
    compilation_provider_status_ = provider.status();
    return;
  }
  compilation_provider_ = *std::move(provider);
}

MusaGpuCompiler::MusaGpuCompiler(
    std::unique_ptr<musa::MusaCompilationProvider> compilation_provider)
    : GpuCompiler(stream_executor::musa::kMusaPlatformId, musa::TargetTriple(),
                  musa::DataLayout()),
      compilation_provider_(std::move(compilation_provider)),
      compilation_provider_status_(
          compilation_provider_ != nullptr
              ? absl::OkStatus()
              : absl::InvalidArgumentError(
                    "MUSA compilation provider must not be null")) {}

se::Platform::Id MusaGpuCompiler::PlatformId() const {
  return stream_executor::musa::kMusaPlatformId;
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

void MusaGpuCompiler::AddGemmRewriterPasses(
    HloPassPipeline& pipeline, const DebugOptions& debug_options,
    const se::GpuComputeCapability& gpu_version,
    const se::SemanticVersion& toolkit_version) {
  // The muBLAS route is deliberately narrower than CUDA/ROCm: it does not
  // expose FP8, bias fusion, or a workspace output. GemmRewriter enforces the
  // qualified type and epilogue subset before emitting the MUSA-specific
  // custom-call target; the shared autotuner subsequently writes a normalized
  // algorithm identifier into its backend config.
  (void)debug_options;
  pipeline.AddPass<DotAlgorithmRewriter>();
  pipeline.AddPass<GemmRewriter>(
      gpu_version, toolkit_version,
      GemmRewriterOptions{GemmRewriterOptions::DType::kNonFp8Only,
                          GemmRewriterOptions::BiasMode::kNoBias});
}

absl::Status MusaGpuCompiler::OptimizeHloConvolutionCanonicalization(
    HloModule* hlo_module, const se::GpuComputeCapability& gpu_version,
    se::dnn::VersionInfo dnn_version,
    const se::SemanticVersion& toolkit_version,
    CompilationStats* compilation_stats) {
  // Convolutions remain on the generic HLO path until an optional muDNN
  // adapter and its canonicalization contract are available.
  (void)hlo_module;
  (void)gpu_version;
  (void)dnn_version;
  (void)toolkit_version;
  (void)compilation_stats;
  return absl::OkStatus();
}

absl::Status MusaGpuCompiler::AddAutotunerPass(
    HloPassPipeline* pipeline, HloModule* hlo_module,
    const se::GpuComputeCapability& gpu_version, const CompileOptions& options,
    tsl::thread::ThreadPool* thread_pool, se::StreamExecutor* stream_executor,
    const GpuTargetConfig* target_config, const AliasInfo* alias_info,
    mlir::MLIRContext* mlir_context,
    HloCostAnalysis::ShapeSizeFunction shape_size_fn,
    const MultiProcessKeyValueStore& key_value_store) {
  return GpuCompiler::AddAutotunerPass(
      pipeline, hlo_module, gpu_version, options, thread_pool, stream_executor,
      target_config, alias_info, mlir_context, std::move(shape_size_fn),
      key_value_store);
}

absl::StatusOr<GpuCompiler::BackendCompileResult>
MusaGpuCompiler::CompileTargetBinary(
    const HloModuleConfig& module_config, llvm::Module* llvm_module,
    const stream_executor::DeviceDescription& device_description,
    bool relocatable, const HloModule* debug_module,
    std::optional<int> shard_number) {
  (void)shard_number;
  if (!compilation_provider_status_.ok()) {
    return ProviderUnavailable(compilation_provider_status_);
  }
  if (compilation_provider_ == nullptr) {
    return absl::FailedPreconditionError(
        "MUSA compilation provider was not initialized");
  }
  if (llvm_module == nullptr) {
    return absl::InvalidArgumentError("MUSA LLVM module must not be null");
  }
  if (!device_description.gpu_compute_capability().IsMusa()) {
    return absl::InvalidArgumentError(
        "MUSA compiler received a non-MUSA device description");
  }
  if (relocatable) {
    return absl::UnimplementedError(
        "MUSA compilation provider does not support relocatable objects");
  }
  // Shared GPU codegen asks the target compiler to process a constants
  // module before compiling individual kernels. When no LLVM globals were
  // emitted this module is intentionally empty and has no target binary.
  if (llvm_module->empty() && llvm_module->global_empty()) {
    return BackendCompileResult{};
  }

  const std::string module_name = SafeMusaModuleName(
      debug_module != nullptr ? debug_module->name()
                              : llvm_module->getName().str());
  absl::StatusOr<musa::MusaLlvm14CompatibilityResult> compatible =
      musa::NormalizeMusaLlvmForLlvm14(*llvm_module, module_name);
  if (!compatible.ok()) return compatible.status();

  ASSIGN_OR_RETURN(musa::MusaCompilationOptions options,
                   GetMusaCompilationOptions(module_config));

  absl::StatusOr<musa::MusaCompilationArtifact> artifact =
      compilation_provider_->Compile(*compatible, options);
  if (!artifact.ok()) return artifact.status();
  return BackendCompileResult{
      .asm_text = {},
      .binary = std::move(artifact->mubin),
      .module_stats = {},
  };
}

absl::StatusOr<stream_executor::ExecutableAbiVersion>
MusaGpuCompiler::CreateExecutableAbiVersion(
    const HloModule& module,
    const stream_executor::DeviceDescription& device_description,
    absl::Span<const uint8_t> main_binary) const {
  if (!compilation_provider_status_.ok()) {
    return ProviderUnavailable(compilation_provider_status_);
  }
  if (compilation_provider_ == nullptr) {
    return absl::FailedPreconditionError(
        "MUSA compilation provider was not initialized");
  }
  ASSIGN_OR_RETURN(musa::MusaCompilationOptions options,
                   GetMusaCompilationOptions(module.config()));
  std::vector<musa::MusaOptionalLibraryAbi> required_optional_libraries =
      RequiredMusaOptionalLibraries(module);
  return musa::BuildMusaExecutableEnvelope(
      device_description, compilation_provider_->identity(),
      compilation_provider_->capabilities(), options, main_binary,
      required_optional_libraries);
}

bool MusaGpuCompiler::UseAotCompiledThunks(const HloModule& module) const {
  (void)module;
  // MUSA deserialization must never re-enter LLVM or the isolated compiler
  // bridge. Always select the self-contained compiled-thunk representation,
  // even if a caller carries an old explicit false value for the experiment.
  return true;
}

absl::Status MusaGpuCompiler::ValidatePersistentKernelCache(
    const HloModuleConfig& module_config) const {
  if (!module_config.debug_options().xla_gpu_kernel_cache_file().empty()) {
    return absl::FailedPreconditionError(
        "MUSA persistent GPU kernel cache is disabled until the shared cache "
        "format records the complete MUSA architecture, provider, bridge, "
        "toolchain, libdevice, shim, and numerical identity");
  }
  return absl::OkStatus();
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
