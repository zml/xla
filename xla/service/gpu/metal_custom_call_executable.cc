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

#include "xla/service/gpu/metal_custom_call_executable.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/ffi/attribute_map.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/execution_state.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/ffi/ffi_registry.h"
#include "xla/ffi/invoke.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/types.h"

namespace xla {
namespace gpu {
namespace {

struct MetalCustomCallOperand {
  int64_t parameter_number = -1;
  Shape shape;
};

struct MetalCustomCallResult {
  ShapeIndex index;
  Shape shape;
};

struct MetalCustomCallConfig {
  std::string target_name;
  std::vector<MetalCustomCallOperand> operands;
  std::vector<MetalCustomCallResult> results;
  ffi::AttributesMap attributes;
};

bool IsSupportedCustomCallBufferShape(const Shape& shape) {
  return shape.IsArray() && !shape.IsOpaque() && !shape.IsToken();
}

absl::StatusOr<ffi::AttributesMap> ParseAttributes(
    const HloCustomCallInstruction& custom_call) {
  auto backend_config = custom_call.backend_config<GpuBackendConfig>();
  if (!backend_config.ok()) {
    VLOG(3) << "Unable to parse Metal custom-call backend config: "
            << backend_config.status().message()
            << "; falling back to raw backend config string.";
  }

  std::string attributes =
      backend_config.ok()
          ? backend_config->custom_call_backend_config().attributes()
          : custom_call.raw_backend_config_string();
  if (attributes.empty()) {
    return ffi::AttributesMap();
  }

  mlir::MLIRContext context;
  mlir::Attribute attr = mlir::parseAttribute(attributes, &context);
  auto dict = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(attr);
  if (dict == nullptr) {
    return absl::InvalidArgumentError(
        "Metal FFI custom-call backend config must be an MLIR dictionary "
        "attribute.");
  }
  return ffi::BuildAttributesMap(dict);
}

absl::StatusOr<MetalCustomCallConfig> MatchMetalCustomCall(
    const HloModule& module) {
  const HloInstruction* root =
      module.entry_computation()->root_instruction();
  if (root->opcode() != HloOpcode::kCustomCall) {
    return absl::UnimplementedError(
        "Metal FFI custom-call executable requires a custom-call root.");
  }

  const auto& custom_call = *Cast<HloCustomCallInstruction>(root);
  if (custom_call.api_version() !=
      CustomCallApiVersion::API_VERSION_TYPED_FFI) {
    return absl::UnimplementedError(
        "Metal custom calls currently require API_VERSION_TYPED_FFI.");
  }
  if (!custom_call.called_computations().empty()) {
    return absl::UnimplementedError(
        "Metal custom calls with called computations are not supported yet.");
  }
  if (!custom_call.output_operand_aliasing().empty()) {
    return absl::UnimplementedError(
        "Metal custom-call output-to-operand aliasing is not supported yet.");
  }

  MetalCustomCallConfig config;
  config.target_name = custom_call.custom_call_target();
  TF_ASSIGN_OR_RETURN(config.attributes, ParseAttributes(custom_call));

  for (const HloInstruction* operand : custom_call.operands()) {
    if (operand->opcode() != HloOpcode::kParameter) {
      return absl::UnimplementedError(
          "Metal FFI custom-call operands must be entry parameters.");
    }
    if (!IsSupportedCustomCallBufferShape(operand->shape())) {
      return absl::UnimplementedError(
          "Metal FFI custom-call operands must be array buffers.");
    }
    config.operands.push_back(
        MetalCustomCallOperand{operand->parameter_number(), operand->shape()});
  }

  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      custom_call.shape(), [&](const Shape& subshape,
                               const ShapeIndex& index) {
        if (subshape.IsToken()) {
          return absl::UnimplementedError(
              "Metal FFI custom-call token results are not supported yet.");
        }
        if (!subshape.IsArray()) {
          return absl::OkStatus();
        }
        if (!IsSupportedCustomCallBufferShape(subshape)) {
          return absl::UnimplementedError(
              "Metal FFI custom-call results must be array buffers.");
        }
        config.results.push_back(MetalCustomCallResult{index, subshape});
        return absl::OkStatus();
      }));

  if (config.results.empty()) {
    return absl::UnimplementedError(
        "Metal FFI custom calls must return at least one array buffer.");
  }

  return config;
}

ffi::CallFrame BuildCallFrame(
    const MetalCustomCallConfig& config,
    const std::vector<se::DeviceAddressBase>& arguments,
    const std::vector<se::DeviceAddressBase>& results) {
  ffi::CallFrameBuilder builder(arguments.size(), results.size());
  for (int64_t i = 0; i < arguments.size(); ++i) {
    builder.AddBufferArg(arguments[i], config.operands[i].shape.element_type(),
                         config.operands[i].shape.dimensions());
  }
  for (int64_t i = 0; i < results.size(); ++i) {
    builder.AddBufferRet(results[i], config.results[i].shape.element_type(),
                         config.results[i].shape.dimensions());
  }
  if (!config.attributes.empty()) {
    ffi::CallFrameBuilder::AttributesBuilder attrs;
    attrs.Append(config.attributes);
    builder.AddAttributes(attrs.Build());
  }
  return builder.Build();
}

std::vector<se::DeviceAddressBase> PlaceholderArguments(
    const MetalCustomCallConfig& config) {
  std::vector<se::DeviceAddressBase> arguments;
  arguments.reserve(config.operands.size());
  for (const MetalCustomCallOperand& operand : config.operands) {
    arguments.push_back(
        se::DeviceAddressBase(nullptr, ShapeUtil::ByteSizeOf(operand.shape, 8)));
  }
  return arguments;
}

std::vector<se::DeviceAddressBase> PlaceholderResults(
    const MetalCustomCallConfig& config) {
  std::vector<se::DeviceAddressBase> results;
  results.reserve(config.results.size());
  for (const MetalCustomCallResult& result : config.results) {
    results.push_back(
        se::DeviceAddressBase(nullptr, ShapeUtil::ByteSizeOf(result.shape, 8)));
  }
  return results;
}

absl::Status InstantiateIfNeeded(const MetalCustomCallConfig& config,
                                 XLA_FFI_Handler_Bundle bundle,
                                 ffi::ExecutionState* execution_state) {
  if (bundle.instantiate == nullptr) {
    return absl::OkStatus();
  }
  std::vector<se::DeviceAddressBase> arguments = PlaceholderArguments(config);
  std::vector<se::DeviceAddressBase> results = PlaceholderResults(config);
  ffi::CallFrame call_frame = BuildCallFrame(config, arguments, results);
  ffi::InvokeContext context;
  context.state_context.instantiate = execution_state;
  return ffi::Invoke(ffi::GetXlaFfiApi(), bundle.instantiate, call_frame,
                     context, XLA_FFI_ExecutionStage_INSTANTIATE);
}

struct OutputAllocation {
  ShapeIndex index;
  se::ScopedDeviceAddress<uint8_t> buffer;
};

class MetalCustomCallExecutable final : public Executable {
 public:
  MetalCustomCallExecutable(std::shared_ptr<HloModule> module,
                            MetalCustomCallConfig config,
                            XLA_FFI_Handler_Bundle bundle,
                            std::unique_ptr<ffi::ExecutionState> execution_state)
      : Executable(std::move(module)),
        config_(std::move(config)),
        result_shape_(this->module().output_shape()),
        bundle_(bundle),
        execution_state_(std::move(execution_state)) {}

  Shape result_shape() const override { return result_shape_; }

  absl::StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments) override {
    se::Stream* stream = run_options->stream();
    if (stream == nullptr) {
      return absl::InvalidArgumentError("Metal custom call requires a stream.");
    }
    se::StreamExecutor* executor = stream->parent();
    se::DeviceAddressAllocator* allocator = run_options->allocator();
    if (allocator == nullptr) {
      return absl::InvalidArgumentError(
          "Metal custom call requires an allocator.");
    }

    const int device_ordinal = run_options->device_ordinal() != -1
                                   ? run_options->device_ordinal()
                                   : executor->device_ordinal();

    std::vector<se::DeviceAddressBase> arg_addresses;
    arg_addresses.reserve(config_.operands.size());
    for (const MetalCustomCallOperand& operand : config_.operands) {
      if (operand.parameter_number < 0 ||
          operand.parameter_number >= arguments.size()) {
        return absl::InvalidArgumentError(
            "Metal custom-call parameter number is out of bounds.");
      }
      se::DeviceAddressBase address =
          arguments[operand.parameter_number].Buffer({}).AsDeviceAddress();
      if (address.is_null()) {
        return absl::InvalidArgumentError(
            "Metal custom-call input buffer is null.");
      }
      arg_addresses.push_back(address);
    }

    std::vector<OutputAllocation> output_allocations;
    std::vector<se::DeviceAddressBase> result_addresses;
    output_allocations.reserve(config_.results.size());
    result_addresses.reserve(config_.results.size());
    for (const MetalCustomCallResult& result : config_.results) {
      const int64_t output_size = ShapeUtil::ByteSizeOf(result.shape, 8);
      TF_ASSIGN_OR_RETURN(se::ScopedDeviceAddress<uint8_t> output_buffer,
                          allocator->Allocate(device_ordinal, output_size));
      result_addresses.push_back(*output_buffer);
      output_allocations.push_back(
          OutputAllocation{result.index, std::move(output_buffer)});
    }

    ffi::CallFrame call_frame =
        BuildCallFrame(config_, arg_addresses, result_addresses);
    ffi::InvokeContext context;
    context.run_id = run_options->run_options().run_id();
    context.device_ordinal = device_ordinal;
    context.backend_context = ffi::InvokeContext::GpuContext{
        /*stream=*/stream,
        /*allocator=*/allocator,
    };
    context.state_context.instantiate = execution_state_.get();

    TF_RETURN_IF_ERROR(ffi::Invoke(ffi::GetXlaFfiApi(), bundle_.execute,
                                   call_frame, context,
                                   XLA_FFI_ExecutionStage_EXECUTE));

    ExecutionOutput output(result_shape_, allocator, device_ordinal,
                           executor->device_ordinal());
    ShapedBuffer* shaped_result =
        static_cast<ShapedBuffer*>(output.MutableResult());
    for (OutputAllocation& allocation : output_allocations) {
      shaped_result->set_buffer(allocation.buffer.Release(),
                                allocation.index);
    }
    output.Commit();
    return std::move(output);
  }

 private:
  MetalCustomCallConfig config_;
  Shape result_shape_;
  XLA_FFI_Handler_Bundle bundle_;
  std::unique_ptr<ffi::ExecutionState> execution_state_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<Executable>> BuildMetalCustomCallExecutable(
    std::shared_ptr<HloModule> module) {
  TF_ASSIGN_OR_RETURN(MetalCustomCallConfig config,
                      MatchMetalCustomCall(*module));
  TF_ASSIGN_OR_RETURN(ffi::HandlerRegistration registration,
                      ffi::FindHandler(config.target_name, "METAL"));
  if (registration.bundle.execute == nullptr) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Metal FFI custom call '%s' has no execute handler.",
        config.target_name));
  }
  if (registration.bundle.prepare != nullptr ||
      registration.bundle.initialize != nullptr) {
    return absl::UnimplementedError(
        "Metal FFI custom calls do not support prepare/initialize handlers "
        "yet.");
  }

  auto execution_state = std::make_unique<ffi::ExecutionState>();
  TF_RETURN_IF_ERROR(
      InstantiateIfNeeded(config, registration.bundle, execution_state.get()));

  return std::make_unique<MetalCustomCallExecutable>(
      std::move(module), std::move(config), registration.bundle,
      std::move(execution_state));
}

}  // namespace gpu
}  // namespace xla
