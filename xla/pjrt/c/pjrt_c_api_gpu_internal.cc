/* Copyright 2023 The OpenXLA Authors.

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

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "google/protobuf/text_format.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/backends/profiler/plugin/plugin_tracer_impl.h"
#include "xla/backends/profiler/plugin/profiler_c_api.h"
#include "xla/backends/profiler/plugin/profiler_error.h"
#include "xla/client/local_client.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/ffi.h"
#include "xla/pjrt/common_pjrt_client.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_abi_version_extension.h"
#include "xla/pjrt/c/pjrt_c_api_custom_partitioner_extension.h"
#include "xla/pjrt/c/pjrt_c_api_execute_chain_extension.h"
#include "xla/pjrt/c/pjrt_c_api_ffi_extension.h"
#include "xla/pjrt/c/pjrt_c_api_ffi_internal.h"
#include "xla/pjrt/c/pjrt_c_api_gpu_extension.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_layouts_extension.h"
#include "xla/pjrt/c/pjrt_c_api_memory_descriptions_extension.h"
#include "xla/pjrt/c/pjrt_c_api_multi_slice_internal_types.h"
#include "xla/pjrt/c/pjrt_c_api_profiler_extension.h"
#include "xla/pjrt/c/pjrt_c_api_shardings_extension.h"
#include "xla/pjrt/c/pjrt_c_api_status_utils.h"
#include "xla/pjrt/c/pjrt_c_api_stream_extension.h"
#include "xla/pjrt/c/pjrt_c_api_triton_extension.h"
#include "xla/pjrt/c/pjrt_c_api_triton_internal.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/c/pjrt_c_api_xla_transform_extension.h"
#include "xla/pjrt/c/pjrt_c_api_xla_transform_internal.h"
#include "xla/pjrt/extensions/abi_version/gpu_abi_version_extension.h"
#include "xla/pjrt/extensions/cross_host_transfers/pjrt_c_api_cross_host_transfers_extension.h"
#include "xla/pjrt/gpu/gpu_helpers.h"
#include "xla/pjrt/gpu/se_gpu_topology_description.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_pjrt_client.h"
#include "xla/python/custom_call_batch_partitioner.h"
#include "xla/python/custom_partition_callback.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/gpu_topology.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#endif  // GOOGLE_CUDA

namespace pjrt {
namespace gpu_plugin {

#if TENSORFLOW_USE_ROCM
#define PJRT_GPU_PLUGIN_PLATFORM_NAME "ROCM"
#elif TENSORFLOW_USE_SYCL
#define PJRT_GPU_PLUGIN_PLATFORM_NAME "ONEAPI"
#else
#define PJRT_GPU_PLUGIN_PLATFORM_NAME "CUDA"
#endif

const PJRT_Api* GetGpuPjrtApi();

namespace {

absl::Status ConvertExecuteChainOptions(PJRT_ExecuteOptions* c_options,
                                        xla::ExecuteOptions* options,
                                        std::shared_ptr<xla::ExecuteContext>*
                                            execute_context) {
  if (c_options == nullptr) {
    return absl::InvalidArgumentError(
        "PJRT_LoadedExecutable_ExecuteChain requires non-null options.");
  }
  absl::Status struct_size_status = ActualStructSizeIsGreaterOrEqual(
      "PJRT_ExecuteOptions",
      PJRT_STRUCT_SIZE(PJRT_ExecuteOptions, incarnation_ids),
      c_options->struct_size);
  if (!struct_size_status.ok()) {
    return struct_size_status;
  }

  if (c_options->num_send_ops > 0 || c_options->num_recv_ops > 0) {
    return absl::UnimplementedError(
        "PJRT_LoadedExecutable_ExecuteChain does not support send/recv "
        "callbacks.");
  }
  if (c_options->struct_size >=
          PJRT_STRUCT_SIZE(PJRT_ExecuteOptions,
                           num_hlo_output_callbacks) &&
      c_options->num_hlo_output_callbacks > 0) {
    return absl::UnimplementedError(
        "PJRT_LoadedExecutable_ExecuteChain does not support HLO output "
        "callbacks.");
  }

  options->launch_id = c_options->launch_id;
  if (c_options->call_location) {
    options->call_location = std::string(c_options->call_location);
  }
  if (c_options->context) {
    *execute_context = c_options->context->execute_context;
  }
  options->context = execute_context->get();
  options->multi_slice_config = nullptr;
  if (c_options->struct_size >= PJRT_ExecuteOptions_STRUCT_SIZE &&
      c_options->multi_slice_config != nullptr) {
    options->multi_slice_config = c_options->multi_slice_config->config.get();
  }
  options->use_major_to_minor_data_layout_for_callbacks = true;
  if (c_options->struct_size >=
      PJRT_STRUCT_SIZE(PJRT_ExecuteOptions,
                       use_major_to_minor_data_layout_for_callbacks)) {
    options->use_major_to_minor_data_layout_for_callbacks =
        c_options->use_major_to_minor_data_layout_for_callbacks;
  }
  for (int i = 0; i < c_options->num_non_donatable_input_indices; ++i) {
    options->non_donatable_input_indices.insert(
        c_options->non_donatable_input_indices[i]);
  }
  for (size_t i = 0; i < c_options->num_tasks; ++i) {
    options->incarnations.insert(
        {c_options->task_ids[i],
         xla::IncarnationId(c_options->incarnation_ids[i])});
  }
  return absl::OkStatus();
}

absl::Status CheckExecuteChainIndex(size_t value, absl::string_view name) {
  if (value > std::numeric_limits<int>::max()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s is too large for ExecuteChain.", name));
  }
  return absl::OkStatus();
}

}  // namespace

PJRT_Error* PJRT_LoadedExecutable_ExecuteChain(
    PJRT_LoadedExecutable_ExecuteChain_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_LoadedExecutable_ExecuteChain_Args",
      PJRT_LoadedExecutable_ExecuteChain_Args_STRUCT_SIZE, args->struct_size));

  xla::ExecuteOptions options;
  std::shared_ptr<xla::ExecuteContext> execute_context;
  PJRT_RETURN_IF_ERROR(
      ConvertExecuteChainOptions(args->options, &options, &execute_context));

  if (args->num_steps == 0) {
    return StatusToPjRtError(absl::InvalidArgumentError(
        "PJRT_LoadedExecutable_ExecuteChain requires at least one step."));
  }
  if (args->steps == nullptr) {
    return StatusToPjRtError(absl::InvalidArgumentError(
        "PJRT_LoadedExecutable_ExecuteChain requires non-null steps."));
  }
  if (args->num_devices == 0) {
    return StatusToPjRtError(absl::InvalidArgumentError(
        "PJRT_LoadedExecutable_ExecuteChain requires at least one device."));
  }
  PJRT_RETURN_IF_ERROR(CheckExecuteChainIndex(args->num_steps, "num_steps"));
  PJRT_RETURN_IF_ERROR(CheckExecuteChainIndex(args->num_devices, "num_devices"));

  std::vector<xla::CommonPjRtLoadedExecutable::ExecuteChainStep> chain_steps;
  chain_steps.reserve(args->num_steps);
  for (size_t step_index = 0; step_index < args->num_steps; ++step_index) {
    PJRT_ExecuteChain_Step* c_step = &args->steps[step_index];
    PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
        "PJRT_ExecuteChain_Step", PJRT_ExecuteChain_Step_STRUCT_SIZE,
        c_step->struct_size));
    if (c_step->executable == nullptr) {
      return StatusToPjRtError(absl::InvalidArgumentError(
          absl::StrFormat("ExecuteChain step %d has a null executable.",
                          step_index)));
    }
    const auto* executable =
        dynamic_cast<const xla::CommonPjRtLoadedExecutable*>(
            c_step->executable->get());
    if (executable == nullptr) {
      return StatusToPjRtError(absl::UnimplementedError(
          absl::StrFormat("ExecuteChain step %d is not a common PJRT loaded "
                          "executable.",
                          step_index)));
    }
    if (c_step->num_args > 0 && c_step->argument_refs == nullptr) {
      return StatusToPjRtError(absl::InvalidArgumentError(absl::StrFormat(
          "ExecuteChain step %d has null argument_refs.", step_index)));
    }
    if (c_step->num_outputs > 0 && c_step->returned_outputs == nullptr) {
      return StatusToPjRtError(absl::InvalidArgumentError(absl::StrFormat(
          "ExecuteChain step %d has null returned_outputs.", step_index)));
    }
    PJRT_RETURN_IF_ERROR(CheckExecuteChainIndex(c_step->num_args, "num_args"));
    PJRT_RETURN_IF_ERROR(
        CheckExecuteChainIndex(c_step->num_outputs, "num_outputs"));

    xla::CommonPjRtLoadedExecutable::ExecuteChainStep step;
    step.executable = executable;
    step.arguments.resize(args->num_devices);
    for (size_t device_index = 0; device_index < args->num_devices;
         ++device_index) {
      step.arguments[device_index].reserve(c_step->num_args);
      for (size_t arg_index = 0; arg_index < c_step->num_args; ++arg_index) {
        PJRT_ExecuteChain_Input* c_input =
            &c_step
                 ->argument_refs[device_index * c_step->num_args + arg_index];
        PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
            "PJRT_ExecuteChain_Input", PJRT_ExecuteChain_Input_STRUCT_SIZE,
            c_input->struct_size));

        xla::CommonPjRtLoadedExecutable::ExecuteChainArgument argument;
        switch (c_input->kind) {
          case PJRT_ExecuteChain_InputKind_Buffer:
            if (c_input->buffer == nullptr) {
              return StatusToPjRtError(absl::InvalidArgumentError(
                  absl::StrFormat("ExecuteChain step %d argument %d on device "
                                  "%d is a null buffer.",
                                  step_index, arg_index, device_index)));
            }
            argument.kind = xla::CommonPjRtLoadedExecutable::
                ExecuteChainArgument::Kind::kBuffer;
            argument.buffer = c_input->buffer->buffer.get();
            break;
          case PJRT_ExecuteChain_InputKind_Output:
            PJRT_RETURN_IF_ERROR(
                CheckExecuteChainIndex(c_input->output_step, "output_step"));
            PJRT_RETURN_IF_ERROR(
                CheckExecuteChainIndex(c_input->output_index, "output_index"));
            argument.kind = xla::CommonPjRtLoadedExecutable::
                ExecuteChainArgument::Kind::kOutput;
            argument.output_step = static_cast<int>(c_input->output_step);
            argument.output_index = static_cast<int>(c_input->output_index);
            break;
          default:
            return StatusToPjRtError(absl::InvalidArgumentError(
                absl::StrFormat("ExecuteChain step %d argument %d on device %d "
                                "has invalid input kind %d.",
                                step_index, arg_index, device_index,
                                c_input->kind)));
        }
        step.arguments[device_index].push_back(argument);
      }
    }
    step.returned_outputs.assign(c_step->returned_outputs,
                                 c_step->returned_outputs + c_step->num_outputs);
    chain_steps.push_back(std::move(step));
  }

  const bool fill_futures =
      args->device_complete_events != nullptr || execute_context != nullptr;
  PJRT_ASSIGN_OR_RETURN(
      xla::CommonPjRtLoadedExecutable::ExecuteChainResult result,
      xla::CommonPjRtLoadedExecutable::ExecuteChain(chain_steps, options,
                                                    fill_futures));

  for (size_t step_index = 0; step_index < args->num_steps; ++step_index) {
    PJRT_ExecuteChain_Step* c_step = &args->steps[step_index];
    for (size_t output_index = 0; output_index < c_step->num_outputs;
         ++output_index) {
      if (!c_step->returned_outputs[output_index]) {
        continue;
      }
      if (c_step->output_lists == nullptr) {
        return StatusToPjRtError(absl::InvalidArgumentError(absl::StrFormat(
            "ExecuteChain step %d returns output %d but output_lists is null.",
            step_index, output_index)));
      }
      for (size_t device_index = 0; device_index < args->num_devices;
           ++device_index) {
        if (c_step->output_lists[device_index] == nullptr) {
          return StatusToPjRtError(absl::InvalidArgumentError(absl::StrFormat(
              "ExecuteChain step %d output list for device %d is null.",
              step_index, device_index)));
        }
        if (step_index >= result.outputs.size() ||
            device_index >= result.outputs[step_index].size() ||
            output_index >= result.outputs[step_index][device_index].size() ||
            result.outputs[step_index][device_index][output_index] == nullptr) {
          return StatusToPjRtError(absl::InternalError(absl::StrFormat(
              "ExecuteChain step %d output %d for device %d was not produced.",
              step_index, output_index, device_index)));
        }
        c_step->output_lists[device_index][output_index] = new PJRT_Buffer{
            std::move(result.outputs[step_index][device_index][output_index]),
            c_step->executable->client};
      }
    }
  }

  if (fill_futures) {
    if (!result.futures.has_value() ||
        result.futures->size() != args->num_devices) {
      return StatusToPjRtError(absl::InternalError(
          "ExecuteChain did not return the expected completion futures."));
    }
    if (execute_context != nullptr) {
      for (xla::Future<>& future : *result.futures) {
        future.OnReady([execute_context](absl::Status status) {});
      }
    }
    if (args->device_complete_events != nullptr) {
      for (size_t device_index = 0; device_index < args->num_devices;
           ++device_index) {
        args->device_complete_events[device_index] =
            new PJRT_Event{std::move((*result.futures)[device_index])};
      }
    }
  }

  return nullptr;
}

PJRT_Error* PJRT_Client_Create(PJRT_Client_Create_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Client_Create_Args", PJRT_Client_Create_Args_STRUCT_SIZE,
      args->struct_size));

  absl::flat_hash_map<std::string, xla::PjRtValueType> create_options =
      pjrt::ConvertFromPjRtNamedValueList(args->create_options,
                                          args->num_options);
  const auto kExpectedOptionNameAndTypes =
      absl::flat_hash_map<std::string, PJRT_NamedValue_Type>({
          {"platform_name", PJRT_NamedValue_Type::PJRT_NamedValue_kString},
          {"allocator", PJRT_NamedValue_Type::PJRT_NamedValue_kString},
          {"memory_fraction", PJRT_NamedValue_Type::PJRT_NamedValue_kFloat},
          {"preallocate", PJRT_NamedValue_Type::PJRT_NamedValue_kBool},
          {"collective_memory_size",
           PJRT_NamedValue_Type::PJRT_NamedValue_kInt64},
          {"visible_devices", PJRT_NamedValue_Type::PJRT_NamedValue_kInt64List},
          {"node_id", PJRT_NamedValue_Type::PJRT_NamedValue_kInt64},
          {"num_nodes", PJRT_NamedValue_Type::PJRT_NamedValue_kInt64},
          {"should_stage_host_to_device_transfers",
           PJRT_NamedValue_Type::PJRT_NamedValue_kBool},
          {"abort_collectives_on_failure",
           PJRT_NamedValue_Type::PJRT_NamedValue_kBool},
          {"use_tfrt_gpu_client", PJRT_NamedValue_Type::PJRT_NamedValue_kBool},
          {"enable_mock_nccl", PJRT_NamedValue_Type::PJRT_NamedValue_kBool},
          {"mock_gpu_topology", PJRT_NamedValue_Type::PJRT_NamedValue_kString},
          {"partition_index", PJRT_NamedValue_Type::PJRT_NamedValue_kInt64},
      });
  PJRT_RETURN_IF_ERROR(
      ValidateCreateOptions(create_options, kExpectedOptionNameAndTypes));

  std::optional<std::string> platform_name;
  if (auto it = create_options.find("platform_name");
      it != create_options.end()) {
    platform_name.emplace(std::get<std::string>(it->second));
  }
  xla::GpuAllocatorConfig allocator_config;
  if (auto it = create_options.find("allocator"); it != create_options.end()) {
    auto allocator_name = std::get<std::string>(it->second);
    if (allocator_name == "default") {
      allocator_config.kind = xla::GpuAllocatorConfig::Kind::kDefault;
    } else if (allocator_name == "platform") {
      allocator_config.kind = xla::GpuAllocatorConfig::Kind::kPlatform;
    } else if (allocator_name == "bfc") {
      allocator_config.kind = xla::GpuAllocatorConfig::Kind::kBFC;
    } else if (allocator_name == "cuda_async") {
      allocator_config.kind = xla::GpuAllocatorConfig::Kind::kCudaAsync;
    } else if (allocator_name == "vmm") {
      allocator_config.kind = xla::GpuAllocatorConfig::Kind::kVmm;
    } else if (allocator_name == "address") {
      allocator_config.kind = xla::GpuAllocatorConfig::Kind::kAddress;
    } else {
      return StatusToPjRtError(absl::UnimplementedError(absl::StrFormat(
          "Allocator %s not supported for PJRT GPU plugin. Supported "
          "allocator "
          "options are: 'default', 'platform', 'bfc', 'cuda_async', 'vmm' and "
          "'address'.",
          allocator_name)));
    }
  }
  if (auto it = create_options.find("memory_fraction");
      it != create_options.end()) {
    allocator_config.memory_fraction = std::get<float>(it->second);
  }
  if (auto it = create_options.find("preallocate");
      it != create_options.end()) {
    allocator_config.preallocate = std::get<bool>(it->second);
  }
  if (auto it = create_options.find("collective_memory_size");
      it != create_options.end()) {
    allocator_config.collective_memory_size = std::get<int64_t>(it->second);
  }
  std::optional<std::set<int>> visible_devices;
  if (auto it = create_options.find("visible_devices");
      it != create_options.end()) {
    const auto& vec = std::get<std::vector<int64_t>>(it->second);
    visible_devices.emplace(vec.begin(), vec.end());
  }
  int node_id = 0;
  if (auto it = create_options.find("node_id"); it != create_options.end()) {
    node_id = std::get<int64_t>(it->second);
  }
  int num_nodes = 1;
  if (auto it = create_options.find("num_nodes"); it != create_options.end()) {
    num_nodes = std::get<int64_t>(it->second);
  }
  bool should_stage_host_to_device_transfers = true;
  if (auto it = create_options.find("should_stage_host_to_device_transfers");
      it != create_options.end()) {
    should_stage_host_to_device_transfers = std::get<bool>(it->second);
  }
  bool abort_collectives_on_failure = false;
  if (auto it = create_options.find("abort_collectives_on_failure");
      it != create_options.end()) {
    abort_collectives_on_failure = std::get<bool>(it->second);
  }
  bool use_tfrt_gpu_client = false;
  if (auto it = create_options.find("use_tfrt_gpu_client");
      it != create_options.end()) {
    use_tfrt_gpu_client = std::get<bool>(it->second);
  }
  bool enable_mock_nccl = false;
  if (auto it = create_options.find("enable_mock_nccl");
      it != create_options.end()) {
    enable_mock_nccl = std::get<bool>(it->second);
  }
  std::optional<std::string> mock_gpu_topology;
  if (auto it = create_options.find("mock_gpu_topology");
      it != create_options.end()) {
    mock_gpu_topology = std::get<std::string>(it->second);
  }
  std::optional<int64_t> partition_index;
  if (auto it = create_options.find("partition_index");
      it != create_options.end()) {
    partition_index = std::get<int64_t>(it->second);
  }

  xla::GpuClientOptions options;
  options.allocator_config = allocator_config;
  options.node_id = node_id;
  options.num_nodes = num_nodes;
  options.allowed_devices = visible_devices;
  options.platform_name = platform_name;
  options.kv_store = pjrt::ToCppKeyValueStore(
      args->kv_get_callback, args->kv_get_user_arg, args->kv_try_get_callback,
      args->kv_try_get_user_arg, args->kv_put_callback, args->kv_put_user_arg);
  options.should_stage_host_to_device_transfers =
      should_stage_host_to_device_transfers;
  options.abort_collectives_on_failure = abort_collectives_on_failure;
  options.use_tfrt_gpu_client = use_tfrt_gpu_client;
  options.enable_mock_nccl = enable_mock_nccl;
  options.mock_gpu_topology = mock_gpu_topology;
  options.partition_index = partition_index;
  PJRT_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> client,
                        xla::GetXlaPjrtGpuClient(options));
  args->client = pjrt::CreateWrapperClient(GetGpuPjrtApi(), std::move(client));
  return nullptr;
}

PJRT_Error* PJRT_ExecuteContext_Create(PJRT_ExecuteContext_Create_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_ExecuteContext_Create_Args",
      PJRT_ExecuteContext_Create_Args_STRUCT_SIZE, args->struct_size));
  auto execute_context = std::make_unique<xla::ExecuteContext>();
  args->context = pjrt::CreateWrapperExecuteContext(std::move(execute_context));
  return nullptr;
}

namespace {

struct TargetConfigAndDevices {
  stream_executor::GpuTargetConfigProto target_config_proto;
  xla::cpu::TargetMachineOptionsProto host_target_machine_options;
  std::vector<int> device_ids;
};

// Parses the 'target_config' entry in 'options'. The option is
// parsed as GpuTargetConfigProto. If there is no 'target_config' in
// 'options', the function falls back to creating a local client,
// returning the local client's target config.
absl::StatusOr<TargetConfigAndDevices> GetTargetConfigFromOptions(
    const absl::flat_hash_map<std::string, xla::PjRtValueType>& options) {
  std::optional<stream_executor::GpuTargetConfigProto> target_config_proto;

  if (auto target_config_it = options.find("target_config");
      target_config_it != options.end()) {
    std::string target_config_proto_string =
        std::get<std::string>(target_config_it->second);
    if (!tsl::protobuf::TextFormat::ParseFromString(
            target_config_proto_string, &target_config_proto.emplace())) {
      return absl::FailedPreconditionError(
          "Failed to parse GpuTargetConfigProto "
          "from the 'target_config' parameter.");
    }
  }

  std::optional<xla::cpu::TargetMachineOptionsProto>
      host_target_machine_options;
  if (auto host_target_machine_options_it =
          options.find("host_target_machine_options");
      host_target_machine_options_it != options.end()) {
    std::string host_target_machine_options_proto_string =
        std::get<std::string>(host_target_machine_options_it->second);
    if (!tsl::protobuf::TextFormat::ParseFromString(
            host_target_machine_options_proto_string,
            &host_target_machine_options.emplace())) {
      return absl::FailedPreconditionError(
          "Failed to parse TargetMachineOptions "
          "from the 'host_target_machine_options' parameter.");
    }
  }
  if (!host_target_machine_options.has_value()) {
    host_target_machine_options.emplace(
        xla::cpu::TargetMachineOptions().ToProto());
  }
  if (target_config_proto.has_value()) {
    return {{*target_config_proto, *host_target_machine_options, {}}};
  }
  ASSIGN_OR_RETURN(xla::LocalClient * xla_client,
                   xla::GetGpuXlaClient(/*platform_name=*/std::nullopt,
                                        /*allowed_devices=*/std::nullopt));
  stream_executor::StreamExecutor* executor =
      xla_client->backend().default_stream_executor();
  std::vector<int> device_ids;
  device_ids.reserve(xla_client->backend().stream_executors().size());
  for (stream_executor::StreamExecutor* executor :
       xla_client->backend().stream_executors()) {
    device_ids.push_back(executor->device_ordinal());
  }
  auto gpu_target_config = xla::Compiler::GpuTargetConfig(executor);
  return {
      {gpu_target_config.ToProto(), *host_target_machine_options, device_ids}};
}

}  // namespace

PJRT_Error* PJRT_GpuDeviceTopology_Create(
    PJRT_TopologyDescription_Create_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_TopologyDescription_Create_Args",
      PJRT_TopologyDescription_Create_Args_STRUCT_SIZE, args->struct_size));

  // Determine the platform ID and name based on the platform.
  xla::PjRtPlatformId platform_id;
  std::string platform_name;
  absl::string_view plugin_platform = PJRT_GPU_PLUGIN_PLATFORM_NAME;

  if (plugin_platform == "ROCM") {
    platform_id = xla::RocmId();
    platform_name = xla::RocmName();
  } else if (plugin_platform == "ONEAPI") {
    platform_id = xla::OneapiId();
    platform_name = xla::OneapiName();
  } else {
    platform_id = xla::CudaId();
    platform_name = xla::CudaName();
  }

  absl::flat_hash_map<std::string, xla::PjRtValueType> create_options =
      pjrt::ConvertFromPjRtNamedValueList(args->create_options,
                                          args->num_options);

  PJRT_ASSIGN_OR_RETURN(TargetConfigAndDevices target_config_and_devices,
                        GetTargetConfigFromOptions(create_options));

  std::vector<int>& device_ids = target_config_and_devices.device_ids;
  stream_executor::GpuTargetConfigProto& target_config_proto =
      target_config_and_devices.target_config_proto;
  xla::TopologySizes sizes{1, 1, static_cast<int>(device_ids.size())};

  if (auto topology_it = create_options.find("topology");
      topology_it != create_options.end()) {
    std::string topology_string = std::get<std::string>(topology_it->second);
    PJRT_ASSIGN_OR_RETURN(sizes,
                          xla::TopologySizes::FromString(topology_string));
  }

  if (sizes.GetDeviceCount() == 0) {
    // If the user did not specify the topology and we did not
    // get any devices from the client, then error out because
    // we do not know how many devices the topology should have.
    return StatusToPjRtError(
        absl::FailedPreconditionError("Cannot create topology without an "
                                      "explicit topology shape or without "
                                      "a client"));
  }

  if (sizes.GetDeviceCount() != device_ids.size()) {
    device_ids.resize(sizes.GetDeviceCount());
    absl::c_iota(device_ids, 0);
  }

  PJRT_ASSIGN_OR_RETURN(
      auto gpu_target_config,
      xla::Compiler::GpuTargetConfig::FromProto(target_config_proto));

  PJRT_ASSIGN_OR_RETURN(
      auto host_target_machine_options,
      xla::cpu::TargetMachineOptions::FromProto(
          target_config_and_devices.host_target_machine_options));

  auto gpu_topology = std::make_shared<const xla::GpuTopology>(
      target_config_proto.device_description_str(), sizes.num_partitions,
      sizes.num_hosts_per_partition, sizes.num_devices_per_host,
      std::move(gpu_target_config), std::move(host_target_machine_options));

  std::string target_config_attr;
  if (!tsl::protobuf::TextFormat::PrintToString(target_config_proto,
                                                &target_config_attr)) {
    return StatusToPjRtError(
        absl::FailedPreconditionError("Cannot serialize target_config_proto"));
  }
  std::string host_target_machine_options_attr;
  if (!tsl::protobuf::TextFormat::PrintToString(
          target_config_and_devices.host_target_machine_options,
          &host_target_machine_options_attr)) {
    return StatusToPjRtError(absl::FailedPreconditionError(
        "Cannot serialize host_target_machine_options"));
  }
  auto pjrt_topology =
      std::make_unique<xla::StreamExecutorGpuTopologyDescription>(
          platform_id, platform_name, std::move(gpu_topology),
          absl::flat_hash_map<std::string, xla::PjRtDeviceAttribute>{
              {"target_config", std::move(target_config_attr)},
              {"host_target_machine_options",
               std::move(host_target_machine_options_attr)}},
          std::move(target_config_proto));
  args->topology = CreateWrapperDeviceTopology(std::move(pjrt_topology));
  return nullptr;
}

#if GOOGLE_CUDA && defined(CUDART_VERSION)  // cuda
namespace {

const std::vector<PJRT_NamedValue>* MakeCudaPluginCAttributes() {
  std::vector<PJRT_NamedValue>* attributes = new std::vector<PJRT_NamedValue>();
  const std::vector<PJRT_NamedValue>& base_attributes =
      pjrt::GetXlaPluginCAttributes();
  attributes->reserve(base_attributes.size() + 1);
  attributes->assign(base_attributes.begin(), base_attributes.end());
  {
    // Include the cuda_version attribute.
    PJRT_NamedValue c_value;
    c_value.struct_size = PJRT_NamedValue_STRUCT_SIZE;
    c_value.extension_start = nullptr;
    absl::string_view name = "cuda_version";
    c_value.name = name.data();
    c_value.name_size = name.size();
    c_value.type = PJRT_NamedValue_Type::PJRT_NamedValue_kInt64;
    c_value.int64_value = CUDART_VERSION;
    c_value.value_size = 1;
    attributes->push_back(c_value);
  }
  return attributes;
}

}  // namespace
#endif

PJRT_Error* PJRT_Plugin_Attributes_Gpu(PJRT_Plugin_Attributes_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Plugin_Attributes_Args", PJRT_Plugin_Attributes_Args_STRUCT_SIZE,
      args->struct_size));
#if GOOGLE_CUDA && defined(CUDART_VERSION)  // cuda
  static const std::vector<PJRT_NamedValue>* attributes =
      MakeCudaPluginCAttributes();
#else
  const std::vector<PJRT_NamedValue>* attributes =
      &pjrt::GetXlaPluginCAttributes();
#endif
  args->num_attributes = attributes->size();
  args->attributes = attributes->data();
  return nullptr;
}

PLUGIN_Profiler_Api profiler_api{
    /*struct_size=*/PLUGIN_Profiler_Api_STRUCT_SIZE,
    /*priv=*/nullptr,
    /*error_destroy=*/xla::profiler::PLUGIN_Profiler_Error_Destroy,
    /*error_message=*/xla::profiler::PLUGIN_Profiler_Error_Message,
    /*error_get_code=*/xla::profiler::PLUGIN_Profiler_Error_GetCode,
    /*create=*/xla::profiler::PLUGIN_Profiler_Create,
    /*destroy=*/xla::profiler::PLUGIN_Profiler_Destroy,
    /*start=*/xla::profiler::PLUGIN_Profiler_Start,
    /*stop=*/xla::profiler::PLUGIN_Profiler_Stop,
    /*collect_data=*/xla::profiler::PLUGIN_Profiler_CollectData,
};

PJRT_Profiler_Extension profiler_extension{
    PJRT_Extension_Base{
        /*struct_size=*/PJRT_Profiler_Extension_STRUCT_SIZE,
        /*type=*/PJRT_Extension_Type::PJRT_Extension_Type_Profiler,
        /*next=*/nullptr,
    },
    /*profiler_api=*/&profiler_api,
};

PJRT_Error* PJRT_Register_Custom_Partitioner(
    PJRT_Register_Custom_Partitioner_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Register_Custom_Partitioner_Args",
      PJRT_Register_Custom_Partitioner_Args_STRUCT_SIZE, args->struct_size));
  std::string name(args->name, args->name_size);
  RegisterCustomCallPartitioner(
      name, jax::CreateCApiCustomCallPartitioner(args->callbacks));
  return nullptr;
}

PJRT_Error* PJRT_Register_Batch_Partitionable(
    PJRT_Register_Batch_Partitionable_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Register_Batch_Partitionable_Args",
      PJRT_Register_Batch_Partitionable_Args_STRUCT_SIZE, args->struct_size));
  std::string name(args->name, args->name_size);
  RegisterCustomCallPartitioner(
      name, std::make_unique<xla::CustomCallBatchPartitioner>());
  return nullptr;
}

PJRT_Custom_Partitioner_Extension custom_partitioner{
    PJRT_Extension_Base{
        /*struct_size=*/PJRT_Custom_Partitioner_Extension_STRUCT_SIZE,
        /*type=*/PJRT_Extension_Type::PJRT_Extension_Type_Custom_Partitioner,
        /*next=*/&profiler_extension.base,
    },
    /*register_custom_partitioner=*/PJRT_Register_Custom_Partitioner,
    /*register_batch_partitionable=*/PJRT_Register_Batch_Partitionable,
};

PJRT_Error* PJRT_Get_Stream_For_External_Ready_Events(
    PJRT_Get_Stream_For_External_Ready_Events_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Get_Stream_For_External_Ready_Events_Args",
      PJRT_Get_Stream_For_External_Ready_Events_Args_STRUCT_SIZE,
      args->struct_size));
  PJRT_ASSIGN_OR_RETURN(
      args->stream, args->device->device->GetStreamForExternalReadyEvents());
  return nullptr;
}

PJRT_Error* PJRT_Wait_Until_Buffer_Ready_On_Stream(
    PJRT_Wait_Until_Buffer_Ready_On_Stream_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Wait_Until_Buffer_Ready_On_Stream_Args",
      PJRT_Wait_Until_Buffer_Ready_On_Stream_Args_STRUCT_SIZE,
      args->struct_size));
  PJRT_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::PjRtBuffer::ExternalReference> external_reference,
      args->buffer->buffer->AcquireExternalReference());
  PJRT_RETURN_IF_ERROR(
      external_reference->WaitUntilBufferReadyOnStream(args->stream));
  return nullptr;
}

PJRT_Stream_Extension stream{
    PJRT_Extension_Base{
        /*struct_size=*/PJRT_Stream_Extension_STRUCT_SIZE,
        /*type=*/PJRT_Extension_Type::PJRT_Extension_Type_Stream,
        /*next=*/&custom_partitioner.base,
    },
    /*get_stream=*/PJRT_Get_Stream_For_External_Ready_Events,
    /*wait_stream=*/PJRT_Wait_Until_Buffer_Ready_On_Stream,
};

PJRT_Error* PJRT_Gpu_Register_Custom_Call(
    PJRT_Gpu_Register_Custom_Call_Args* args) {
  PJRT_RETURN_IF_ERROR(ActualStructSizeIsGreaterOrEqual(
      "PJRT_Gpu_Register_Custom_Call_Args",
      PJRT_Gpu_Register_Custom_Call_Args_STRUCT_SIZE, args->struct_size));
  std::string function_name(args->function_name, args->function_name_size);
  switch (args->api_version) {
    case 0:
      xla::CustomCallTargetRegistry::Global()->Register(
          function_name, args->handler_execute, PJRT_GPU_PLUGIN_PLATFORM_NAME);
      return nullptr;
    case 1:
      xla::ffi::Ffi::RegisterStaticHandler(
          xla::ffi::GetXlaFfiApi(), function_name,
          PJRT_GPU_PLUGIN_PLATFORM_NAME,
          XLA_FFI_Handler_Bundle{
              reinterpret_cast<XLA_FFI_Handler*>(args->handler_instantiate),
              reinterpret_cast<XLA_FFI_Handler*>(args->handler_prepare),
              reinterpret_cast<XLA_FFI_Handler*>(args->handler_initialize),
              reinterpret_cast<XLA_FFI_Handler*>(args->handler_execute)});
      return nullptr;
    default:
      return StatusToPjRtError(absl::UnimplementedError(
          absl::StrFormat("API version %d not supported for PJRT GPU plugin. "
                          "Supported versions are 0 and 1.",
                          args->api_version)));
  }
}

const PJRT_Api* GetGpuPjrtApi() {
  static PJRT_Gpu_Custom_Call custom_call{
      PJRT_Extension_Base{
          /*struct_size=*/PJRT_Gpu_Custom_Call_STRUCT_SIZE,
          /*type=*/PJRT_Extension_Type::PJRT_Extension_Type_Gpu_Custom_Call,
          /*next=*/&stream.base,
      },
      /*custom_call=*/PJRT_Gpu_Register_Custom_Call,
  };

  static PJRT_Layouts_Extension layouts_extension =
      pjrt::CreateLayoutsExtension(&custom_call.base);

  static PJRT_FFI_Extension ffi_extension =
      pjrt::CreateFfiExtension(&layouts_extension.base);

  static PJRT_MemoryDescriptions_Extension memory_descriptions_extension =
      pjrt::CreateMemoryDescriptionsExtension(&ffi_extension.base);

  static PJRT_Triton_Extension triton_extension =
      pjrt::CreateTritonExtension(&memory_descriptions_extension.base);

  static PJRT_CrossHostTransfers_Extension cross_host_transfers_extension =
      pjrt::CreateCrossHostTransfersExtension(&triton_extension.base);

  static PJRT_Shardings_Extension shardings_extension =
      pjrt::CreateShardingsExtension(&cross_host_transfers_extension.base);

  static PJRT_Xla_Transform_Extension xla_transform_extension =
      pjrt::CreateXlaTransformExtension(&shardings_extension.base);

  static PJRT_AbiVersion_Extension abi_version_extension =
      pjrt::CreateGpuAbiVersionExtension(&xla_transform_extension.base);

  static PJRT_ExecuteChain_Extension execute_chain_extension{
      PJRT_Extension_Base{
          /*struct_size=*/PJRT_ExecuteChain_Extension_STRUCT_SIZE,
          /*type=*/PJRT_Extension_Type::PJRT_Extension_Type_ExecuteChain,
          /*next=*/&abi_version_extension.base,
      },
      /*execute_chain=*/PJRT_LoadedExecutable_ExecuteChain,
  };

  static const PJRT_Api pjrt_api = pjrt::CreatePjrtApi(
      pjrt::gpu_plugin::PJRT_Client_Create,
      pjrt::gpu_plugin::PJRT_ExecuteContext_Create,
      pjrt::gpu_plugin::PJRT_GpuDeviceTopology_Create,
      pjrt::PJRT_Plugin_Initialize_NoOp, &execute_chain_extension.base,
      pjrt::gpu_plugin::PJRT_Plugin_Attributes_Gpu);

  return &pjrt_api;
}

}  // namespace gpu_plugin
}  // namespace pjrt
