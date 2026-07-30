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

#include "xla/backends/gpu/autotuner/mudnn.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/gpu_conv_runner.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/engine_options.h"
#include "xla/stream_executor/musa/musa_compute_capability.h"
#include "xla/stream_executor/musa/musa_mudnn_api.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

namespace se = ::stream_executor;
using MudnnBackendConfig = se::dnn::AlgorithmProto;

class OwningScratchAllocator final : public se::ScratchAllocator {
 public:
  OwningScratchAllocator(int device_ordinal,
                         se::DeviceAddressAllocator* allocator)
      : device_ordinal_(device_ordinal), allocator_(allocator) {}

  int64_t GetMemoryLimitInBytes() override { return -1; }

  absl::StatusOr<se::DeviceAddress<uint8_t>> AllocateBytes(
      int64_t byte_size) override {
    if (byte_size < 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("byte_size must be non-negative, but got ", byte_size));
    }
    ASSIGN_OR_RETURN(se::ScopedDeviceAddress<uint8_t> buffer,
                     allocator_->Allocate(device_ordinal_, byte_size,
                                          /*retry_on_failure=*/false));
    buffers_.push_back(std::move(buffer));
    return *buffers_.back();
  }

 private:
  int device_ordinal_;
  se::DeviceAddressAllocator* allocator_;
  absl::InlinedVector<se::ScopedDeviceAddress<uint8_t>, 4> buffers_;
};

bool IsUnfusedDnnConvolution(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kCustomCall) {
    return false;
  }
  const absl::string_view target = instr.custom_call_target();
  return target == kCudnnConvForwardCallTarget ||
         target == kCudnnConvBackwardInputCallTarget ||
         target == kCudnnConvBackwardFilterCallTarget;
}

absl::Status ValidateAlgorithmConfig(const MudnnBackendConfig& config) {
  if (config.algo_id() != XLA_MUSA_MUDNN_ALGORITHM_IMPLICIT_GEMM &&
      config.algo_id() != XLA_MUSA_MUDNN_ALGORITHM_WINOGRAD_NONFUSED &&
      config.algo_id() != XLA_MUSA_MUDNN_ALGORITHM_GEMM) {
    return absl::InvalidArgumentError(
        absl::StrCat("normalized muDNN algorithm must be one of 0, 2, or 3; got ",
                     config.algo_id()));
  }
  if (!config.tuning_knobs().empty()) {
    return absl::InvalidArgumentError(
        "normalized muDNN algorithms do not accept tuning knobs");
  }
  if (config.is_cudnn_frontend()) {
    return absl::InvalidArgumentError(
        "normalized muDNN algorithms do not accept cuDNN frontend state");
  }
  if (!config.has_workspace_size()) {
    return absl::InvalidArgumentError(
        "normalized muDNN algorithms require an explicit workspace size");
  }
  if (config.workspace_size().value() >
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return absl::InvalidArgumentError(
        "muDNN workspace does not fit in an XLA shape dimension");
  }
  return absl::OkStatus();
}

// Replaces the custom call when the selected algorithm requires workspace.
// ConvRewriter initially creates (result, u8[0]); this preserves that shared
// ABI while resizing only the final scratch element.
absl::Status ApplyConfigAndUpdateWorkspaceInOutputTuple(
    HloInstruction& instr, const MudnnBackendConfig& config) {
  if (!instr.shape().IsTuple() || instr.shape().tuple_shapes().size() < 2) {
    return absl::InvalidArgumentError(
        "muDNN convolution custom call must return (result, workspace)");
  }

  HloComputation* computation = instr.parent();
  absl::InlinedVector<Shape, 2> new_call_element_shapes;
  for (int i = 0; i < instr.shape().tuple_shapes().size() - 1; ++i) {
    new_call_element_shapes.push_back(instr.shape().tuple_shapes(i));
  }
  const int64_t workspace_size =
      static_cast<int64_t>(config.workspace_size().value());
  new_call_element_shapes.push_back(ShapeUtil::MakeShape(U8, {workspace_size}));
  Shape new_call_shape = ShapeUtil::MakeTupleShape(new_call_element_shapes);

  HloInstruction* new_call = computation->AddInstruction(
      instr.CloneWithNewOperands(new_call_shape, instr.operands()));
  new_call->SetAndSanitizeName(instr.name());

  ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                   instr.backend_config<GpuBackendConfig>());
  *gpu_config.mutable_cudnn_conv_backend_config()->mutable_algorithm() = config;
  RETURN_IF_ERROR(new_call->set_backend_config(std::move(gpu_config)));

  absl::InlinedVector<HloInstruction*, 2> new_tuple_elements;
  for (int i = 0; i < new_call->shape().tuple_shapes().size() - 1; ++i) {
    new_tuple_elements.push_back(
        computation->AddInstruction(HloInstruction::CreateGetTupleElement(
            new_call->shape().tuple_shapes(i), new_call, i)));
  }
  new_tuple_elements.push_back(computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR1<uint8_t>({}))));
  HloInstruction* replacement = computation->AddInstruction(
      HloInstruction::CreateTuple(new_tuple_elements));
  return computation->ReplaceInstruction(&instr, replacement);
}

absl::Status ApplyConfigToMudnnCustomCall(HloInstruction& instr,
                                          const MudnnBackendConfig& config) {
  RETURN_IF_ERROR(ValidateAlgorithmConfig(config));
  if (config.workspace_size().value() > 0) {
    return ApplyConfigAndUpdateWorkspaceInOutputTuple(instr, config);
  }
  ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                   instr.backend_config<GpuBackendConfig>());
  *gpu_config.mutable_cudnn_conv_backend_config()->mutable_algorithm() = config;
  return instr.set_backend_config(std::move(gpu_config));
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
GetConvolutionConfigs(const HloCustomCallInstruction& instr,
                      se::StreamExecutor* stream_executor,
                      se::DeviceAddressAllocator* allocator) {
  if (stream_executor == nullptr || allocator == nullptr) {
    return absl::FailedPreconditionError(
        "muDNN autotuning requires a live MUSA executor and allocator");
  }
  se::dnn::DnnSupport* dnn = stream_executor->AsDnn();
  if (dnn == nullptr) {
    return absl::FailedPreconditionError(
        "muDNN autotuning requires initialized DNN support");
  }

  ASSIGN_OR_RETURN(GpuConvConfig gpu_conv_config, GetGpuConvConfig(&instr));
  const se::dnn::ConvolutionKind conv_kind =
      CudnnConvKindToProto(gpu_conv_config.kind);
  ASSIGN_OR_RETURN(se::dnn::DataType input_type,
                   GetDNNDataTypeFromPrimitiveType(gpu_conv_config.input_type));
  ASSIGN_OR_RETURN(
      se::dnn::DataType output_type,
      GetDNNDataTypeFromPrimitiveType(gpu_conv_config.output_type));
  ASSIGN_OR_RETURN(std::unique_ptr<se::Stream> stream,
                   stream_executor->CreateStream());

  const bool allow_tf32 = absl::c_all_of(
      instr.precision_config().operand_precision(),
      [](int precision) { return precision <= PrecisionConfig::HIGH; });
  const se::EngineOptions engine_options{
      RequireDeterminism(instr.GetModule()->config()), allow_tf32,
      /*require_command_buffer=*/false};
  OwningScratchAllocator scratch_allocator(stream_executor->device_ordinal(),
                                           allocator);

  const auto initialize_buffer = [&stream](se::DeviceAddressBase buffer) {
    return stream->MemZero(&buffer, buffer.size());
  };
  absl::InlinedVector<se::DeviceAddressBase, 2> operand_buffers;
  for (const HloInstruction* operand : instr.operands()) {
    ASSIGN_OR_RETURN(se::DeviceAddress<uint8_t> buffer,
                     scratch_allocator.AllocateBytes(
                         ShapeUtil::ByteSizeOf(operand->shape())));
    RETURN_IF_ERROR(initialize_buffer(buffer));
    operand_buffers.push_back(buffer);
  }

  if (!instr.shape().IsTuple() || instr.shape().tuple_shapes().size() < 2) {
    return absl::InvalidArgumentError(
        "muDNN convolution custom call must return (result, workspace)");
  }
  absl::InlinedVector<se::DeviceAddressBase, 1> result_buffers;
  for (int i = 0; i < instr.shape().tuple_shapes().size() - 1; ++i) {
    ASSIGN_OR_RETURN(se::DeviceAddress<uint8_t> buffer,
                     scratch_allocator.AllocateBytes(
                         ShapeUtil::ByteSizeOf(instr.shape().tuple_shapes(i))));
    result_buffers.push_back(buffer);
  }

  ASSIGN_OR_RETURN(
      GpuConvParams gpu_conv_params,
      GetGpuConvParams(gpu_conv_config, absl::MakeSpan(operand_buffers),
                       absl::MakeSpan(result_buffers)));
  std::vector<std::unique_ptr<const se::dnn::ConvRunner>> runners;
  RETURN_IF_ERROR(dnn->GetConvolveRunners(
      conv_kind, input_type, output_type, stream.get(),
      gpu_conv_config.input_descriptor, gpu_conv_params.input_buf,
      gpu_conv_config.filter_descriptor, gpu_conv_params.filter_buf,
      gpu_conv_config.output_descriptor, gpu_conv_params.output_buf,
      gpu_conv_config.conv_desc,
      /*use_fallback=*/false, &scratch_allocator, engine_options, &runners));

  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.reserve(runners.size());
  for (const std::unique_ptr<const se::dnn::ConvRunner>& runner : runners) {
    ASSIGN_OR_RETURN(se::dnn::AlgorithmDesc descriptor,
                     runner->ToAlgorithmDesc());
    MudnnBackendConfig algorithm = descriptor.ToProto();
    if (algorithm.algo_id() < 0) {
      return absl::FailedPreconditionError(
          absl::StrCat("muDNN returned non-normalized negative algorithm ",
                       algorithm.algo_id()));
    }
    const uint64_t workspace_size = runner->GetWorkspaceSize();
    if (algorithm.has_workspace_size() &&
        algorithm.workspace_size().value() != workspace_size) {
      return absl::FailedPreconditionError(
          "muDNN runner and serialized algorithm disagree on workspace size");
    }
    algorithm.mutable_workspace_size()->set_value(workspace_size);
    auto config = std::make_unique<BackendConfig>();
    *config->mutable_algorithm() = std::move(algorithm);
    configs.push_back(std::move(config));
  }
  return configs;
}

}  // namespace

bool MudnnBackend::IsSupported(const HloInstruction& instr) {
  return IsUnfusedDnnConvolution(instr);
}

absl::StatusOr<std::unique_ptr<BackendConfig>> MudnnBackend::GetDefaultConfig(
    const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return absl::InvalidArgumentError(
        "MudnnBackend requires an unfused DNN convolution custom call");
  }
  ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<BackendConfig>> configs,
      GetConvolutionConfigs(*Cast<const HloCustomCallInstruction>(&instr),
                            stream_executor(), allocator_));
  if (configs.empty()) {
    return absl::FailedPreconditionError(
        "muDNN returned no supported convolution runner for the default "
        "configuration");
  }
  return std::move(configs.front());
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
MudnnBackend::GetSupportedConfigs(const HloInstruction& instr) {
  if (!IsSupported(instr)) {
    return std::vector<std::unique_ptr<BackendConfig>>{};
  }
  if (do_not_autotune_) {
    ASSIGN_OR_RETURN(std::unique_ptr<BackendConfig> default_config,
                     GetDefaultConfig(instr));
    std::vector<std::unique_ptr<BackendConfig>> configs;
    configs.push_back(std::move(default_config));
    return configs;
  }
  return GetConvolutionConfigs(*Cast<const HloCustomCallInstruction>(&instr),
                               stream_executor(), allocator_);
}

absl::Status MudnnBackend::ApplyConfig(HloInstruction& instr,
                                       const BackendConfig& config) {
  if (!IsSupported(instr)) {
    return absl::InvalidArgumentError(
        "MudnnBackend requires an unfused DNN convolution custom call");
  }
  if (!config.has_algorithm()) {
    return absl::InvalidArgumentError(
        "MudnnBackend requires a DNN AlgorithmProto config");
  }
  return ApplyConfigToMudnnCustomCall(instr, config.algorithm());
}

std::string MudnnBackend::version() const {
  const se::DeviceDescription& device = target_config().device_description;
  const se::MusaComputeCapability* capability =
      device.gpu_compute_capability().musa_compute_capability();
  return absl::StrCat(
      "musa_arch=",
      capability == nullptr ? "unknown" : capability->architecture(),
      ";musa_runtime=", device.runtime_version().ToString(),
      ";musa_driver=", device.driver_version().ToString(),
      ";musa_kernel_driver=", device.kernel_mode_driver_version().ToString(),
      ";musa_toolkit=", device.compile_time_toolkit_version().ToString(),
      ";mudnn=", target_config().dnn_version_info.ToString(),
      ";mudnn_required_contract=", se::musa::kMusaMuDnnAbiContractV1,
      ";mudnn_required_fingerprint=", se::musa::kMusaMuDnnAbiFingerprintV1,
      ";mudnn_algorithm_selection=live_runner",
      ";mudnn_deterministic_ops=", debug_options().xla_gpu_deterministic_ops(),
      ";mudnn_exclude_nondeterministic_ops=",
      debug_options().xla_gpu_exclude_nondeterministic_ops());
}

}  // namespace xla::gpu
