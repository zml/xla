/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/autotuner/cudnn.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/transforms/block_scaling_rewriter.h"
#include "xla/backends/gpu/transforms/cudnn_fusion_compiler.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/literal_util.h"
#include "xla/service/algorithm_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/gpu_conv_runner.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/shape.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/engine_options.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace se = ::stream_executor;

using CudnnBackendConfig = stream_executor::dnn::AlgorithmProto;

namespace {

// Replaces the instruction with a new instruction with the same name in the
// parent computation. The given instruction will be replaced by a tuple of the
// convolution result and the workspace size. A few following instructions will
// be added to the parent computation to extract the convolution result from the
// new tuple.
absl::Status ApplyConfigAndUpdateWorkspaceInOutputTuple(
    HloInstruction& instr, const CudnnBackendConfig& config) {
  HloComputation* computation = instr.parent();
  std::vector<Shape> new_call_element_shapes;
  // Add the shapes of the outputs of the convolution.
  new_call_element_shapes.reserve(instr.shape().tuple_shapes().size() - 1);
  for (int i = 0; i < instr.shape().tuple_shapes().size() - 1; ++i) {
    new_call_element_shapes.emplace_back(instr.shape().tuple_shapes(i));
  }
  // The final element is the size of the workspace.
  int64_t workspace_size = config.workspace_size().value();
  new_call_element_shapes.emplace_back(
      ShapeUtil::MakeShape(U8, {workspace_size}));
  Shape new_call_shape = ShapeUtil::MakeTupleShape(new_call_element_shapes);
  HloInstruction* new_call = computation->AddInstruction(
      instr.CloneWithNewOperands(new_call_shape, instr.operands()));
  new_call->SetAndSanitizeName(instr.name());

  ASSIGN_OR_RETURN(GpuBackendConfig gpu_backend_config,
                   instr.backend_config<GpuBackendConfig>());
  CudnnConvBackendConfig* cudnn_conv_config =
      gpu_backend_config.mutable_cudnn_conv_backend_config();
  *cudnn_conv_config->mutable_algorithm() = config;
  RETURN_IF_ERROR(new_call->set_backend_config(gpu_backend_config));

  std::vector<HloInstruction*> new_tuple_elements;
  new_tuple_elements.reserve(new_call->shape().tuple_shapes().size() - 1);
  for (int i = 0; i < new_call->shape().tuple_shapes().size() - 1; ++i) {
    new_tuple_elements.emplace_back(
        computation->AddInstruction(HloInstruction::CreateGetTupleElement(
            new_call->shape().tuple_shapes(i), new_call, i)));
  }
  new_tuple_elements.emplace_back(computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<uint8_t>({}))));

  // Repackage new_call so it has the same shape as the original call, namely
  // (conv_result, u8[0]).
  HloInstruction* new_tuple = computation->AddInstruction(
      HloInstruction::CreateTuple(new_tuple_elements));

  RETURN_IF_ERROR(instr.parent()->ReplaceInstruction(&instr, new_tuple));
  return absl::OkStatus();
}

// Rows of a scaled dot's output, or 0 if it does not have a 2-D-or-more array
// shape. Used to keep sm_103 off cuDNN's thin-M block-scaled kernels.
int64_t ScaledDotOutputRows(const HloInstruction& scaled_dot) {
  const Shape& shape = scaled_dot.shape();
  if (!shape.IsArray() || shape.dimensions().size() < 2) {
    return 0;
  }
  return shape.dimensions(shape.dimensions().size() - 2);
}

bool IsSupportedCudnnFusion(const HloInstruction& instr,
                            se::StreamExecutor* stream_executor,
                            const DebugOptions& debug_options) {
  const HloComputation* computation = instr.fused_instructions_computation();
  const HloInstruction* hero = hlo_query::GetFirstInstructionWithOpcode(
      *computation, {HloOpcode::kDot, HloOpcode::kConvolution,
                     HloOpcode::kScaledDot, HloOpcode::kRaggedDot});
  if (hero == nullptr) {
    VLOG(1) << "Fusion does not contain a dot or convolution.";
    return false;
  }

  PrecisionConfig::Algorithm algorithm = PrecisionConfig::ALG_UNSET;
  if (auto* dot = DynCast<HloDotInstruction>(hero)) {
    algorithm = dot->precision_config().algorithm();
  } else if (auto* conv = DynCast<HloConvolutionInstruction>(hero)) {
    algorithm = conv->precision_config().algorithm();
  } else if (auto* scaled_dot = DynCast<HloScaledDotInstruction>(hero)) {
    algorithm = scaled_dot->precision_config().algorithm();
  } else if (auto* ragged_dot = DynCast<HloRaggedDotInstruction>(hero)) {
    algorithm = ragged_dot->precision_config().algorithm();
  }

  if (!algorithm_util::IsSupportedByCudnn(algorithm)) {
    VLOG(1) << "Fusion contains a precision config not supported by cudnn.";
    return false;
  }

  if (GetDnnVersionInfoOrDefault(stream_executor).major_version() < 9) {
    VLOG(1) << "Cudnn version is too old.";
    return false;
  }

  if (hero->opcode() == HloOpcode::kConvolution ||
      hero->opcode() == HloOpcode::kRaggedDot) {
    return true;
  }

  stream_executor::CudaComputeCapability compute_capability =
      stream_executor->GetDeviceDescription().cuda_compute_capability();

  // Block-scaled graphs stay an allowlist, and a cuDNN upgrade does not by
  // itself put an architecture on it. cuDNN does not decline a block-scaled
  // graph it cannot handle; it fails in whatever way it fails, and two of the
  // three ways are quiet. Measured by forcing this check true and grading an
  // NVFP4 model:
  //
  //   sm_120, 9.22  compiler segfaults in get_heuristics_list, swizzled or not
  //   sm_120, 9.24  output identical to the Triton path            <- allowed
  //   sm_103, 9.24  correct, and the fastest thing there is on the wide MLP
  //                 dots -- but only from 128 output rows up. Below that,
  //                 heuristic 0 still runs while heuristics 1..4 execute an
  //                 illegal instruction and poison the CUDA context, killing
  //                 every later autotuner candidate with them. Nothing is lost
  //                 by declining below it -- Triton beats cuDNN at 16 rows.
  //
  // sm_100 predates all of this and is where the path was first enabled, so it
  // keeps its place. Everything else falls through to Triton until someone has
  // graded its output against a reference. That is a closed allowlist rather
  // than a denylist on purpose: the failure mode here is silent wrong answers,
  // so an unmeasured architecture should get Triton, not a guess.
  //
  // Re-verified 2026-08-24 on two GB300 hosts independently, cuDNN 9.24.0:
  //
  //  * The boundary is exactly 128 and it is a floor, not an alignment
  //    requirement -- 124/126/127 fault, 128/129/130 are clean, and so are
  //    160/192/384. Unchanged across four (K,N) pairs spanning K 4096-19968
  //    and N 6656-39936.
  //  * It is NOT NVFP4-specific. An MXFP8 dot (f8e4m3fn values, e8m0 scales,
  //    group 32) faults at 127 and passes at 128 identically, so this gate is
  //    carrying more than the NVFP4 evidence above claims credit for.
  //  * Below the boundary heuristic 0 alone is correct: it agrees with the
  //    trusted cuBLASLt reference, as do all five above it. No candidate in
  //    ~90 runs was ever demoted for wrong results.
  //
  // Two things that mislead when debugging this, both confirmed by experiment:
  //
  //  * THE FAULT IS ASYNCHRONOUS AND CUDNN NEVER REPORTS IT.
  //    cudnnBackendExecute returns success, CUDNN_LOGLEVEL_DBG=2 logs nothing,
  //    and XLA only discovers it at the stream sync. An earlier version of
  //    this comment attributed it to "err 715 from shimCuLaunchKernelEx"; that
  //    string does not come from this path. 715 is just the numeric value of
  //    CUDA_ERROR_ILLEGAL_INSTRUCTION.
  //  * The follow-on "Failed to load in-memory CUBIN (compiled for a different
  //    GPU?)" on heuristics 2..4 is a RED HERRING. That is the poisoned
  //    context talking, not an arch mismatch -- the same plans build fine at
  //    128 rows. Do not go looking at sm_ targets or cubin packaging.
  //
  // Note this guard can never be validated by anything GetSupportedConfigs
  // returns. cuDNN's heuristic list is 8 plans at 112 rows and 8 at 128; the
  // enumeration path is only validate + build_operation_graph +
  // create_execution_plans + check_support, so nothing is compiled and nothing
  // launched. A pre-emptive row-count guard is the only lever there is.
  //
  // TODO(raph): this is an NVIDIA bug, not ours. Re-test the whole table on
  // the next cuDNN past 9.24 and shrink the gate if thin M starts working;
  // XLA_CUDNN_ALLOW_ANY_BLOCK_SCALED=1 lifts it for exactly that purpose.
  // TODO(raph): cuDNN's block-scaled path is also TN-only, which is not
  // expressed here at all. rhs_contracting_dims={0} (rhs laid out [K,N]) is
  // refused for every dtype at every M with a plain "No valid engine configs",
  // indistinguishable from an ordinary unsupported. Only ={1} (rhs [N,K]) is
  // accepted. Production shapes are [N,K] so nothing is broken today, but a
  // dot emitted the other way round silently never reaches cuDNN -- and
  // tile_ir_test.cc's NVFP4 fixture is ={0}, so it is useless as a cuDNN
  // probe. Worth a real check that says so.
  if (hero->opcode() == HloOpcode::kScaledDot) {
    // XLA_CUDNN_ALLOW_ANY_BLOCK_SCALED=1 lifts the allowlist so a new
    // architecture can be graded; never set it in production.
    static const bool allow_any = [] {
      const char* env = std::getenv("XLA_CUDNN_ALLOW_ANY_BLOCK_SCALED");
      return env != nullptr && absl::string_view(env) == "1";
    }();
    const bool is_sm100 =
        compute_capability.major == se::CudaComputeCapability::kBlackwell &&
        compute_capability.minor == 0;
    const bool has_fixed_cudnn = GetDnnVersionInfoOrDefault(stream_executor) >=
                                 se::dnn::VersionInfo(9, 24, 0);
    const bool is_sm120_with_fixed_cudnn =
        compute_capability.major == se::CudaComputeCapability::kBlackwell_12 &&
        has_fixed_cudnn;
    const bool is_sm103_with_enough_rows =
        compute_capability.major == se::CudaComputeCapability::kBlackwell &&
        compute_capability.minor == 3 && has_fixed_cudnn &&
        ScaledDotOutputRows(*hero) >= 128;
    return allow_any || is_sm100 || is_sm120_with_fixed_cudnn ||
           is_sm103_with_enough_rows;
  }

  if ((compute_capability.IsAtLeastAmpere() &&
       debug_options.xla_gpu_cudnn_gemm_fusion_level() > 1) ||
      (compute_capability.IsAtLeastBlackwell() &&
       debug_options.xla_gpu_cudnn_gemm_fusion_level() > 0)) {
    return true;
  }

  VLOG(1) << "Fusion is not supported by cudnn.";
  return false;
}

absl::StatusOr<std::vector<CudnnBackendConfig>> GetAlgorithms(
    se::dnn::DnnSupport* dnn, se::dnn::ConvolutionKind conv_kind,
    se::dnn::DataType input_type, se::dnn::DataType output_type,
    se::Stream* stream, const GpuConvConfig& gpu_conv_config,
    const se::EngineOptions& engine_options, bool use_fallback) {
  std::vector<std::unique_ptr<const se::dnn::ConvRunner>> conv_runners;
  std::vector<std::unique_ptr<const se::dnn::FusedConvRunner>>
      fused_conv_runners;
  std::vector<std::unique_ptr<const se::dnn::GraphConvRunner>>
      graph_conv_runners;
  switch (conv_kind) {
    case se::dnn::ConvolutionKind::FORWARD_BIAS_ACTIVATION: {
      if (!gpu_conv_config.fusion) {
        return absl::InvalidArgumentError(
            "GpuConvConfig had fusion ConvolutionKind but no FusionConfig.");
      }
      RETURN_IF_ERROR(dnn->GetFusedConvolveRunners(
          se::dnn::ConvolutionKind::FORWARD, input_type,
          BiasTypeForInputType(input_type), output_type,
          gpu_conv_config.conv_result_scale,
          gpu_conv_config.fusion->side_input_scale,
          gpu_conv_config.fusion->leakyrelu_alpha, stream,
          gpu_conv_config.input_descriptor, gpu_conv_config.filter_descriptor,
          gpu_conv_config.bias_descriptor, gpu_conv_config.output_descriptor,
          gpu_conv_config.conv_desc, use_fallback, gpu_conv_config.fusion->mode,
          engine_options, &fused_conv_runners));
      break;
    }
    case se::dnn::ConvolutionKind::FORWARD_GRAPH: {
      RETURN_IF_ERROR(dnn->GetGraphConvolveRunners(
          conv_kind, input_type, output_type, stream,
          gpu_conv_config.input_descriptor, gpu_conv_config.filter_descriptor,
          gpu_conv_config.output_descriptor, gpu_conv_config.conv_desc,
          use_fallback, engine_options, &graph_conv_runners,
          gpu_conv_config.serialized_graph));
      break;
    }
    case se::dnn::ConvolutionKind::FORWARD:
    case se::dnn::ConvolutionKind::BACKWARD_DATA:
    case se::dnn::ConvolutionKind::BACKWARD_FILTER: {
      RETURN_IF_ERROR(dnn->GetConvolveRunners(
          conv_kind, input_type, output_type, stream,
          gpu_conv_config.input_descriptor,
          /*input_data=*/se::DeviceAddressBase(nullptr),
          gpu_conv_config.filter_descriptor,
          /*filter_data=*/se::DeviceAddressBase(nullptr),
          gpu_conv_config.output_descriptor,
          /*output_data=*/se::DeviceAddressBase(nullptr),
          gpu_conv_config.conv_desc, use_fallback,
          /*scratch_allocator=*/nullptr, engine_options, &conv_runners));
      break;
    }
    default:
      return absl::InvalidArgumentError(
          "Cudnn backend doesn't support this convolution kind.");
  }

  std::vector<CudnnBackendConfig> configs;
  if (!conv_runners.empty()) {
    configs.reserve(conv_runners.size());
    for (const auto& runner : conv_runners) {
      configs.push_back(runner->ToAlgorithmDesc()->ToProto());
    }
  } else if (!fused_conv_runners.empty()) {
    configs.reserve(fused_conv_runners.size());
    for (const auto& runner : fused_conv_runners) {
      configs.push_back(runner->ToAlgorithmDesc()->ToProto());
    }
  } else if (!graph_conv_runners.empty()) {
    configs.reserve(graph_conv_runners.size());
    for (const auto& runner : graph_conv_runners) {
      configs.push_back(runner->ToAlgorithmDesc()->ToProto());
    }
  }
  return configs;
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
GetCudnnFusionConfigs(const HloInstruction& instr,
                      se::StreamExecutor* stream_executor,
                      const Compiler::GpuTargetConfig& target_config,
                      const DebugOptions& debug_options) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  bool use_deviceless = false;
  switch (debug_options.xla_gpu_cudnn_deviceless_compilation_mode()) {
    case DebugOptions::CUDNN_DEVICELESS_COMPILATION_UNSET:
    case DebugOptions::CUDNN_DEVICELESS_COMPILATION_DISABLED:
      use_deviceless = false;
      break;
    case DebugOptions::CUDNN_DEVICELESS_COMPILATION_ALWAYS:
      use_deviceless = true;
      break;
    case DebugOptions::CUDNN_DEVICELESS_COMPILATION_AUTO:
    default:
      use_deviceless = (stream_executor == nullptr);
      break;
  }
  if (use_deviceless) {
    if (target_config.dnn_version_info < se::dnn::VersionInfo(9, 8, 0)) {
      return absl::FailedPreconditionError(
          "Deviceless cuDNN compilation requires cuDNN >= 9.8.");
    }
    stream_executor = nullptr;
  } else if (stream_executor == nullptr) {
    return absl::InvalidArgumentError(
        "Null stream executor is not supported when cuDNN deviceless "
        "compilation is disabled.");
  }
  ASSIGN_OR_RETURN(int plan_count,
                   CuDnnFusionCompiler::GetAvailablePlanCount(
                       stream_executor, target_config.device_description,
                       *DynCast<HloFusionInstruction>(&instr)));

  VLOG(2) << "Found " << plan_count << " plans for cudnn fusion.";
  configs.reserve(plan_count);
  for (int plan_id = 0; plan_id < plan_count; ++plan_id) {
    auto config = std::make_unique<BackendConfig>();
    config->mutable_algorithm()->set_algo_id(plan_id);
    configs.push_back(std::move(config));
  }
  return configs;
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
GetConvolutionCustomCallConfigs(const HloCustomCallInstruction* instr,
                                se::StreamExecutor* stream_executor) {
  if (stream_executor == nullptr) {
    return absl::InvalidArgumentError("Null stream executor is not supported.");
  }
  ASSIGN_OR_RETURN(GpuConvConfig gpu_conv_config, GetGpuConvConfig(instr));
  se::dnn::ConvolutionKind conv_kind =
      CudnnConvKindToProto(gpu_conv_config.kind);
  ASSIGN_OR_RETURN(se::dnn::DataType input_type,
                   GetDNNDataTypeFromPrimitiveType(gpu_conv_config.input_type));
  ASSIGN_OR_RETURN(
      se::dnn::DataType output_type,
      GetDNNDataTypeFromPrimitiveType(gpu_conv_config.output_type));
  se::dnn::DnnSupport* dnn = stream_executor->AsDnn();
  auto allocator =
      std::make_unique<stream_executor::StreamExecutorAddressAllocator>(
          stream_executor);
  ASSIGN_OR_RETURN(se::Stream * stream,
                   allocator->GetStream(stream_executor->device_ordinal()));
  bool allow_tf32 = absl::c_all_of(
      instr->precision_config().operand_precision(),
      [](int precision) { return precision <= PrecisionConfig::HIGH; });
  const se::EngineOptions engine_options{
      RequireDeterminism(instr->GetModule()->config()), allow_tf32,
      /*require_command_buffer=*/false};

  // Try to get algorithms without fallback first, as fallback algorithms can be
  // very slow.
  std::vector<CudnnBackendConfig> algorithm_configs;
  ASSIGN_OR_RETURN(
      algorithm_configs,
      GetAlgorithms(dnn, conv_kind, input_type, output_type, stream,
                    gpu_conv_config, engine_options, /*use_fallback=*/false));

  if (algorithm_configs.empty()) {
    ASSIGN_OR_RETURN(
        algorithm_configs,
        GetAlgorithms(dnn, conv_kind, input_type, output_type, stream,
                      gpu_conv_config, engine_options, /*use_fallback=*/true));
  }

  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.reserve(algorithm_configs.size());
  for (const auto& algorithm_config : algorithm_configs) {
    auto config = std::make_unique<BackendConfig>();
    *config->mutable_algorithm() = algorithm_config;
    configs.push_back(std::move(config));
  }
  return configs;
}

absl::Status ApplyConfigToCudnnFusion(HloInstruction& instr,
                                      const CudnnBackendConfig& config) {
  // A block-scaled fusion has to have its scale operands swizzled into the
  // 128x4 layout before it is stamped as a cuDNN fusion, because the graph
  // emitter asserts that layout unconditionally — cudnn_fusion_compiler.cc
  // calls set_reordering_type(F8_128x4) on both scales. Swizzling is what
  // makes the assertion true; without it cuDNN reads every scale factor from
  // the wrong offset and the fusion silently returns garbage rather than
  // failing. The call used to live in GemmFusionAutotuner and was lost with
  // that pass, which is why it is back here and not there.
  HloInstruction* fusion = &instr;
  if (hlo_query::GetFirstInstructionWithOpcode(
          *instr.fused_instructions_computation(), HloOpcode::kScaledDot) !=
      nullptr) {
    ASSIGN_OR_RETURN(fusion, CudnnScaledDotHelper::AddScaleSwizzle(
                                 Cast<HloFusionInstruction>(&instr)));
  }
  ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                   fusion->backend_config<GpuBackendConfig>());
  FusionBackendConfig* backend_config =
      gpu_config.mutable_fusion_backend_config();
  backend_config->set_kind(kCuDnnFusionKind);
  backend_config->mutable_cudnn_fusion_config()->set_plan_id(config.algo_id());
  RETURN_IF_ERROR(fusion->set_backend_config(std::move(gpu_config)));
  return absl::OkStatus();
}

absl::Status ApplyConfigToCudnnCustomCall(HloInstruction& instr,
                                          const CudnnBackendConfig& config) {
  if (config.has_workspace_size() && config.workspace_size().value() > 0) {
    return ApplyConfigAndUpdateWorkspaceInOutputTuple(instr, config);
  }
  ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                   instr.backend_config<GpuBackendConfig>());
  CudnnConvBackendConfig* cudnn_conv_config =
      gpu_config.mutable_cudnn_conv_backend_config();
  *cudnn_conv_config->mutable_algorithm() = config;
  RETURN_IF_ERROR(instr.set_backend_config(std::move(gpu_config)));
  return absl::OkStatus();
}

}  // namespace

bool CudnnBackend::IsSupported(const HloInstruction& instr) {
  if (instr.opcode() == HloOpcode::kFusion) {
    return IsSupportedCudnnFusion(instr, stream_executor(), debug_options());
  }

  if (instr.opcode() == HloOpcode::kCustomCall) {
    return IsCustomCallToDnnConvolution(instr);
  }

  return false;
}

absl::StatusOr<std::unique_ptr<BackendConfig>> CudnnBackend::GetDefaultConfig(
    const HloInstruction& instr) {
  if (IsCustomCallToDnnConvolution(instr)) {
    // If the instruction is a custom call to a DnnConvolution, we can return
    // the default config.
    auto config = std::make_unique<BackendConfig>();
    config->mutable_algorithm()->set_algo_id(-1);
    return config;
  }

  if (stream_executor() != nullptr && instr.opcode() == HloOpcode::kFusion &&
      IsSupportedCudnnFusion(instr, stream_executor(), debug_options())) {
    ASSIGN_OR_RETURN(std::vector<std::unique_ptr<BackendConfig>> configs,
                     GetCudnnFusionConfigs(instr, stream_executor(),
                                           target_config(), debug_options()));
    if (!configs.empty()) {
      return std::move(configs[0]);
    }
  }

  return absl::InvalidArgumentError(
      "Cannot get default config for cudnn backend without device.");
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
CudnnBackend::GetSupportedConfigs(const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return std::vector<std::unique_ptr<BackendConfig>>();
  }
  if (instr.opcode() == HloOpcode::kFusion) {
    return GetCudnnFusionConfigs(instr, stream_executor(), target_config(),
                                 debug_options());
  }
  if (IsCustomCallToDnnConvolution(instr)) {
    auto custom_call_instr = Cast<HloCustomCallInstruction>(&instr);
    return GetConvolutionCustomCallConfigs(custom_call_instr,
                                           stream_executor());
  }

  return std::vector<std::unique_ptr<BackendConfig>>();
}

absl::Status CudnnBackend::ApplyConfig(HloInstruction& instr,
                                       const BackendConfig& config) {
  if (!config.has_algorithm()) {
    return absl::InvalidArgumentError(
        "Expected AlgorithmProto config for CudnnBackend.");
  }
  const CudnnBackendConfig& algorithm_config = config.algorithm();
  if (instr.opcode() == HloOpcode::kFusion) {
    return ApplyConfigToCudnnFusion(instr, algorithm_config);
  }

  if (instr.opcode() == HloOpcode::kCustomCall) {
    return ApplyConfigToCudnnCustomCall(instr, algorithm_config);
  }

  return absl::UnimplementedError(
      "Cudnn backend doesn't support this instruction.");
}

}  // namespace gpu
}  // namespace xla
