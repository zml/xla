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

#include "xla/stream_executor/sycl/sycl_command_buffer.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <sycl/ext/oneapi/experimental/graph.hpp>  // NOLINT
#include <sycl/sycl.hpp>                           // NOLINT

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/bit_pattern.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/sycl/sycl_kernel.h"
#include "xla/stream_executor/sycl/sycl_stream.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::sycl {
namespace {

namespace sycl_exp = ::sycl::ext::oneapi::experimental;
using GraphNode = sycl_exp::node;
using ModifiableGraph =
    sycl_exp::command_graph<sycl_exp::graph_state::modifiable>;
using ExecutableGraph =
    sycl_exp::command_graph<sycl_exp::graph_state::executable>;

std::vector<const CommandBuffer::Command*> CopyDependencies(
    absl::Span<const CommandBuffer::Command* const> dependencies) {
  return std::vector<const CommandBuffer::Command*>(dependencies.begin(),
                                                    dependencies.end());
}

std::string UnsupportedControlFlowMessage(absl::string_view operation) {
  return absl::StrCat(
      "SYCL command buffers do not support ", operation,
      " commands yet: oneAPI SYCL graphs do not currently expose "
      "device-side graph conditionals/loops.");
}

std::string StateName(CommandBuffer::State state) {
  switch (state) {
    case CommandBuffer::State::kCreate:
      return "create";
    case CommandBuffer::State::kUpdate:
      return "update";
    case CommandBuffer::State::kFinalized:
      return "finalized";
  }
}

std::string ModeName(CommandBuffer::Mode mode) {
  switch (mode) {
    case CommandBuffer::Mode::kPrimary:
      return "primary";
    case CommandBuffer::Mode::kNested:
      return "nested";
  }
}

::sycl::nd_range<3> SyclNdRange(const ThreadDim& threads,
                                const BlockDim& blocks) {
  ::sycl::range<3> global_range(blocks.z * threads.z, blocks.y * threads.y,
                                blocks.x * threads.x);
  ::sycl::range<3> local_range(threads.z, threads.y, threads.x);
  return ::sycl::nd_range<3>(global_range, local_range);
}

void AddDependencies(ModifiableGraph& graph, GraphNode& node,
                     const std::vector<GraphNode>& dependencies) {
  for (const GraphNode& dependency : dependencies) {
    GraphNode src = dependency;
    graph.make_edge(src, node);
  }
}

GraphNode AddEmptyNode(ModifiableGraph& graph,
                       const std::vector<GraphNode>& dependencies) {
  GraphNode node = graph.add();
  AddDependencies(graph, node, dependencies);
  return node;
}

int64_t ElapsedMicros(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now() - start)
      .count();
}

}  // namespace

struct SyclCommandBuffer::GraphArtifacts {
  std::unique_ptr<ModifiableGraph> graph;
  std::unique_ptr<ExecutableGraph> executable;
};

struct SyclCommandBuffer::SyclCommand : CommandBuffer::Command {
  enum class Kind { kEmpty, kLaunch, kMemcpyD2D, kMemset, kChild };

  explicit SyclCommand(
      Kind kind, absl::Span<const CommandBuffer::Command* const> deps)
      : kind(kind), dependencies(CopyDependencies(deps)) {}

  virtual std::unique_ptr<SyclCommand> ClonePayload() const = 0;
  virtual std::string Label() const = 0;

  Kind kind;
  std::vector<const CommandBuffer::Command*> dependencies;
  std::optional<GraphNode> graph_node;
};

namespace {

struct LaunchParams {
  ThreadDim threads;
  BlockDim blocks;
  const SyclKernel* kernel = nullptr;
  ::sycl::kernel* function = nullptr;
  std::string kernel_name;
  std::vector<void*> args;
  uint64_t shared_mem_bytes = 0;
};

struct LaunchArgsSnapshot {
  const SyclKernel* kernel = nullptr;
  ::sycl::kernel* function = nullptr;
  std::vector<void*> args;
  uint64_t shared_mem_bytes = 0;
};

struct EmptyCommand final : SyclCommandBuffer::SyclCommand {
  explicit EmptyCommand(absl::Span<const CommandBuffer::Command* const> deps)
      : SyclCommand(Kind::kEmpty, deps) {}

  std::unique_ptr<SyclCommand> ClonePayload() const override {
    return std::make_unique<EmptyCommand>(
        absl::Span<const CommandBuffer::Command* const>());
  }

  std::string Label() const override { return "empty"; }
};

struct LaunchCommand final : SyclCommandBuffer::SyclCommand {
  LaunchCommand(LaunchParams params,
                absl::Span<const CommandBuffer::Command* const> deps)
      : SyclCommand(Kind::kLaunch, deps), params(std::move(params)) {}

  std::unique_ptr<SyclCommand> ClonePayload() const override {
    return std::make_unique<LaunchCommand>(
        params, absl::Span<const CommandBuffer::Command* const>());
  }

  std::string Label() const override {
    return absl::StrCat("launch:", params.kernel_name);
  }

  LaunchParams params;
  std::vector<sycl_exp::dynamic_parameter<void*>> dynamic_args;
  bool node_update_pending = false;
};

struct MemcpyD2DCommand final : SyclCommandBuffer::SyclCommand {
  MemcpyD2DCommand(DeviceAddressBase destination, DeviceAddressBase source,
                   uint64_t size,
                   absl::Span<const CommandBuffer::Command* const> deps)
      : SyclCommand(Kind::kMemcpyD2D, deps),
        destination(destination),
        source(source),
        size(size) {}

  std::unique_ptr<SyclCommand> ClonePayload() const override {
    return std::make_unique<MemcpyD2DCommand>(
        destination, source, size,
        absl::Span<const CommandBuffer::Command* const>());
  }

  std::string Label() const override {
    return absl::StrCat("memcpy_d2d:", size);
  }

  DeviceAddressBase destination;
  DeviceAddressBase source;
  uint64_t size;
};

struct MemsetCommand final : SyclCommandBuffer::SyclCommand {
  MemsetCommand(DeviceAddressBase destination, BitPattern bit_pattern,
                size_t num_elements,
                absl::Span<const CommandBuffer::Command* const> deps)
      : SyclCommand(Kind::kMemset, deps),
        destination(destination),
        bit_pattern(bit_pattern),
        num_elements(num_elements) {}

  std::unique_ptr<SyclCommand> ClonePayload() const override {
    return std::make_unique<MemsetCommand>(
        destination, bit_pattern, num_elements,
        absl::Span<const CommandBuffer::Command* const>());
  }

  std::string Label() const override {
    return absl::StrCat("memset:", bit_pattern.ToString(), "x",
                        num_elements);
  }

  DeviceAddressBase destination;
  BitPattern bit_pattern;
  size_t num_elements;
};

struct ChildCommand final : SyclCommandBuffer::SyclCommand {
  ChildCommand(std::unique_ptr<SyclCommandBuffer> child,
               absl::Span<const CommandBuffer::Command* const> deps)
      : SyclCommand(Kind::kChild, deps), child(std::move(child)) {}

  std::unique_ptr<SyclCommand> ClonePayload() const override {
    absl::StatusOr<std::unique_ptr<SyclCommandBuffer>> cloned =
        child->CloneForChild();
    if (!cloned.ok()) {
      LOG(ERROR) << "Failed to clone SYCL child command buffer: "
                 << cloned.status();
      return nullptr;
    }
    return std::make_unique<ChildCommand>(
        std::move(*cloned), absl::Span<const CommandBuffer::Command* const>());
  }

  std::string Label() const override { return "child"; }

  std::unique_ptr<SyclCommandBuffer> child;
};

absl::StatusOr<LaunchArgsSnapshot> SnapshotLaunchArgs(const Kernel& kernel,
                                                      const KernelArgs& args) {
  const auto* sycl_kernel = dynamic_cast<const SyclKernel*>(&kernel);
  if (sycl_kernel == nullptr) {
    return absl::InvalidArgumentError(
        "SyclCommandBuffer::CreateLaunch requires a SYCL kernel.");
  }

  ::sycl::kernel* function = sycl_kernel->gpu_function();
  if (function == nullptr) {
    return absl::InternalError(
        "SyclCommandBuffer::CreateLaunch: SYCL kernel function is not set.");
  }

  const auto snapshot_packed =
      [&](const KernelArgsPackedArrayBase& packed)
      -> absl::StatusOr<LaunchArgsSnapshot> {
    bool has_shared_memory = packed.number_of_shared_bytes() > 0;
    int32_t expected_number_of_arguments =
        sycl_kernel->Arity() + (has_shared_memory ? 1 : 0);
    if (expected_number_of_arguments != packed.number_of_arguments()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Kernel %s has %d arguments, but expected %d; arity=%d; "
          "has_shared_memory=%s; number_of_shared_bytes=%d",
          kernel.name(), packed.number_of_arguments(),
          expected_number_of_arguments, sycl_kernel->Arity(),
          has_shared_memory ? "true" : "false",
          packed.number_of_shared_bytes()));
    }

    LaunchArgsSnapshot params;
    params.kernel = sycl_kernel;
    params.function = function;
    params.shared_mem_bytes = packed.number_of_shared_bytes();

    params.args.reserve(packed.argument_addresses().size());
    for (const void* const arg : packed.argument_addresses()) {
      params.args.push_back(
          reinterpret_cast<void*>(*static_cast<const uint64_t*>(arg)));
    }

    return params;
  };

  if (auto* packed = DynCast<KernelArgsPackedArrayBase>(&args)) {
    const auto& pack = kernel.args_packing();
    if (!pack || !dynamic_cast<const PackableKernelArgs*>(packed)) {
      return snapshot_packed(*packed);
    }
    ASSIGN_OR_RETURN(auto repacked, pack(kernel, *packed));
    return snapshot_packed(*repacked);
  }

  if (auto* device_mem = DynCast<KernelArgsDeviceAddressArray>(&args)) {
    const auto& pack = kernel.args_packing();
    if (!pack) {
      return absl::InternalError(
          "Kernel is missing a custom arguments packing function for device "
          "memory arguments array.");
    }
    ASSIGN_OR_RETURN(auto packed, pack(kernel, *device_mem));
    return snapshot_packed(*packed);
  }

  return absl::InternalError("Unsupported kernel arguments type.");
}

absl::StatusOr<LaunchParams> SnapshotLaunchParams(
    const ThreadDim& threads, const BlockDim& blocks, const Kernel& kernel,
    const KernelArgs& args) {
  ASSIGN_OR_RETURN(LaunchArgsSnapshot snapshot,
                   SnapshotLaunchArgs(kernel, args));
  LaunchParams params;
  params.threads = threads;
  params.blocks = blocks;
  params.kernel = snapshot.kernel;
  params.function = snapshot.function;
  params.kernel_name = std::string(kernel.name());
  params.args = std::move(snapshot.args);
  params.shared_mem_bytes = snapshot.shared_mem_bytes;
  return params;
}

void RefreshDynamicArgs(LaunchCommand& command) {
  if (command.dynamic_args.size() != command.params.args.size()) {
    command.dynamic_args.clear();
    command.dynamic_args.reserve(command.params.args.size());
    for (void* arg : command.params.args) {
      command.dynamic_args.emplace_back(arg);
    }
    return;
  }
  for (size_t i = 0; i < command.params.args.size(); ++i) {
    command.dynamic_args[i].update(command.params.args[i]);
  }
}

GraphNode AddLaunchNode(ModifiableGraph& graph, LaunchCommand& command,
                        const std::vector<GraphNode>& dependencies) {
  RefreshDynamicArgs(command);
  LaunchParams params = command.params;
  std::vector<sycl_exp::dynamic_parameter<void*>> dynamic_args =
      command.dynamic_args;
  GraphNode node =
      graph.add([params, dynamic_args](::sycl::handler& cgh) mutable {
        ::sycl::nd_range<3> nd_range =
            SyclNdRange(params.threads, params.blocks);
        size_t reflected_total_args =
            dynamic_args.size() +
            (params.shared_mem_bytes > 0 ? size_t{1} : size_t{0});
        try {
          reflected_total_args =
              params.function->get_info<::sycl::info::kernel::num_args>();
        } catch (const ::sycl::exception& e) {
          VLOG(1) << "Failed to reflect argument count for kernel '"
                  << params.kernel_name << "': " << e.what()
                  << "; using packed argument count.";
        }
        for (size_t arg_index = 0; arg_index < dynamic_args.size();
             ++arg_index) {
          cgh.set_arg(arg_index, dynamic_args[arg_index]);
        }
        if (params.shared_mem_bytes > 0) {
          ::sycl::local_accessor<int8_t, 1> local_buffer(
              params.shared_mem_bytes, cgh);
          cgh.set_arg(dynamic_args.size(), local_buffer);
        }
        for (size_t arg_index =
                 dynamic_args.size() +
                 (params.shared_mem_bytes > 0 ? size_t{1} : size_t{0});
             arg_index < reflected_total_args; ++arg_index) {
          void* scratch_arg = nullptr;
          cgh.set_arg(arg_index, scratch_arg);
        }
        cgh.parallel_for(nd_range, *params.function);
      });
  AddDependencies(graph, node, dependencies);
  command.graph_node = node;
  return node;
}

GraphNode AddMemcpyD2DNode(ModifiableGraph& graph,
                           MemcpyD2DCommand& command,
                           const std::vector<GraphNode>& dependencies) {
  if (command.size == 0) {
    command.graph_node = AddEmptyNode(graph, dependencies);
    return *command.graph_node;
  }

  void* destination = const_cast<void*>(command.destination.opaque());
  void* source = const_cast<void*>(command.source.opaque());
  uint64_t size = command.size;
  GraphNode node = graph.add([destination, source, size](::sycl::handler& cgh) {
    cgh.memcpy(destination, source, size);
  });
  AddDependencies(graph, node, dependencies);
  command.graph_node = node;
  return node;
}

GraphNode AddMemsetNode(ModifiableGraph& graph, MemsetCommand& command,
                        const std::vector<GraphNode>& dependencies) {
  if (command.num_elements == 0) {
    command.graph_node = AddEmptyNode(graph, dependencies);
    return *command.graph_node;
  }

  void* destination = const_cast<void*>(command.destination.opaque());
  BitPattern bit_pattern = command.bit_pattern;
  size_t num_elements = command.num_elements;
  GraphNode node =
      graph.add([destination, bit_pattern, num_elements](::sycl::handler& cgh) {
        if (const auto* pattern = std::get_if<uint8_t>(&bit_pattern)) {
          cgh.memset(destination, *pattern, num_elements);
          return;
        }
        if (const auto* pattern = std::get_if<uint16_t>(&bit_pattern)) {
          cgh.fill(static_cast<uint16_t*>(destination), *pattern, num_elements);
          return;
        }
        cgh.fill(static_cast<uint32_t*>(destination),
                 bit_pattern.GetPatternBroadcastedToUint32(), num_elements);
      });
  AddDependencies(graph, node, dependencies);
  command.graph_node = node;
  return node;
}

GraphNode AddExecutableGraphNode(
    ModifiableGraph& graph, const ExecutableGraph& executable,
    const std::vector<GraphNode>& dependencies) {
  GraphNode node = graph.add([executable](::sycl::handler& cgh) {
    cgh.ext_oneapi_graph(executable);
  });
  AddDependencies(graph, node, dependencies);
  return node;
}

absl::StatusOr<std::vector<GraphNode>> AppendCommandsToGraph(
    ModifiableGraph& graph,
    const std::vector<std::unique_ptr<SyclCommandBuffer::SyclCommand>>& commands,
    const std::vector<GraphNode>& dependencies);

absl::StatusOr<std::vector<GraphNode>> AddCommandToGraph(
    ModifiableGraph& graph, SyclCommandBuffer::SyclCommand& command,
    const std::vector<GraphNode>& dependencies) {
  command.graph_node.reset();
  switch (command.kind) {
    case SyclCommandBuffer::SyclCommand::Kind::kEmpty:
      command.graph_node = AddEmptyNode(graph, dependencies);
      return std::vector<GraphNode>{*command.graph_node};

    case SyclCommandBuffer::SyclCommand::Kind::kLaunch:
      return std::vector<GraphNode>{
          AddLaunchNode(graph, static_cast<LaunchCommand&>(command),
                        dependencies),
      };

    case SyclCommandBuffer::SyclCommand::Kind::kMemcpyD2D:
      return std::vector<GraphNode>{
          AddMemcpyD2DNode(graph, static_cast<MemcpyD2DCommand&>(command),
                           dependencies),
      };

    case SyclCommandBuffer::SyclCommand::Kind::kMemset:
      return std::vector<GraphNode>{
          AddMemsetNode(graph, static_cast<MemsetCommand&>(command),
                        dependencies),
      };

    case SyclCommandBuffer::SyclCommand::Kind::kChild: {
      auto& child = static_cast<ChildCommand&>(command);
      if (!child.child->commands().empty()) {
        return AppendCommandsToGraph(graph, child.child->commands(),
                                     dependencies);
      }
      const auto* graph_artifacts = child.child->graph_artifacts();
      if (graph_artifacts != nullptr && graph_artifacts->executable != nullptr) {
        return std::vector<GraphNode>{AddExecutableGraphNode(
            graph, *graph_artifacts->executable, dependencies)};
      }
      return std::vector<GraphNode>{AddEmptyNode(graph, dependencies)};
    }
  }
}

absl::StatusOr<std::vector<GraphNode>> AppendCommandsToGraph(
    ModifiableGraph& graph,
    const std::vector<std::unique_ptr<SyclCommandBuffer::SyclCommand>>&
        commands,
    const std::vector<GraphNode>& dependencies) {
  absl::flat_hash_map<const CommandBuffer::Command*, std::vector<GraphNode>>
      command_nodes;
  absl::flat_hash_set<const CommandBuffer::Command*> referenced_commands;

  for (const auto& command : commands) {
    for (const CommandBuffer::Command* dependency : command->dependencies) {
      referenced_commands.insert(dependency);
    }
  }

  for (const auto& command : commands) {
    std::vector<GraphNode> node_dependencies;
    if (command->dependencies.empty()) {
      node_dependencies = dependencies;
    } else {
      for (const CommandBuffer::Command* dependency : command->dependencies) {
        auto it = command_nodes.find(dependency);
        if (it == command_nodes.end()) {
          return absl::InvalidArgumentError(
              "SYCL command buffer command depends on a command that has not "
              "been added to this graph.");
        }
        node_dependencies.insert(node_dependencies.end(), it->second.begin(),
                                 it->second.end());
      }
    }
    ASSIGN_OR_RETURN(command_nodes[command.get()],
                     AddCommandToGraph(graph, *command, node_dependencies));
  }

  std::vector<GraphNode> leaves;
  for (const auto& command : commands) {
    if (referenced_commands.contains(command.get())) {
      continue;
    }
    auto it = command_nodes.find(command.get());
    if (it != command_nodes.end()) {
      leaves.insert(leaves.end(), it->second.begin(), it->second.end());
    }
  }
  if (leaves.empty()) {
    return dependencies;
  }
  return leaves;
}

std::string CommandKindName(SyclCommandBuffer::SyclCommand::Kind kind) {
  switch (kind) {
    case SyclCommandBuffer::SyclCommand::Kind::kEmpty:
      return "empty";
    case SyclCommandBuffer::SyclCommand::Kind::kLaunch:
      return "launch";
    case SyclCommandBuffer::SyclCommand::Kind::kMemcpyD2D:
      return "memcpy_d2d";
    case SyclCommandBuffer::SyclCommand::Kind::kMemset:
      return "memset";
    case SyclCommandBuffer::SyclCommand::Kind::kChild:
      return "child";
  }
}

size_t CountCommands(
    const std::vector<std::unique_ptr<SyclCommandBuffer::SyclCommand>>&
        commands) {
  size_t count = commands.size();
  for (const auto& command : commands) {
    if (command->kind == SyclCommandBuffer::SyclCommand::Kind::kChild) {
      const auto& child = static_cast<const ChildCommand&>(*command);
      count += CountCommands(child.child->commands());
    }
  }
  return count;
}

void ClearPendingLaunchNodeUpdates(
    const std::vector<std::unique_ptr<SyclCommandBuffer::SyclCommand>>&
        commands) {
  for (const auto& command : commands) {
    if (command->kind == SyclCommandBuffer::SyclCommand::Kind::kLaunch) {
      static_cast<LaunchCommand&>(*command).node_update_pending = false;
    }
    if (command->kind == SyclCommandBuffer::SyclCommand::Kind::kChild) {
      auto& child = static_cast<ChildCommand&>(*command);
      ClearPendingLaunchNodeUpdates(child.child->commands());
    }
  }
}

void CollectPendingLaunchNodes(
    const std::vector<std::unique_ptr<SyclCommandBuffer::SyclCommand>>&
        commands,
    std::vector<GraphNode>* nodes) {
  for (const auto& command : commands) {
    if (command->kind == SyclCommandBuffer::SyclCommand::Kind::kLaunch) {
      auto& launch = static_cast<LaunchCommand&>(*command);
      if (launch.node_update_pending && launch.graph_node.has_value()) {
        launch.graph_node->update_nd_range(
            SyclNdRange(launch.params.threads, launch.params.blocks));
        nodes->push_back(*launch.graph_node);
      }
    }
    if (command->kind == SyclCommandBuffer::SyclCommand::Kind::kChild) {
      auto& child = static_cast<ChildCommand&>(*command);
      CollectPendingLaunchNodes(child.child->commands(), nodes);
    }
  }
}

}  // namespace

absl::StatusOr<std::unique_ptr<CommandBuffer>> SyclCommandBuffer::Create(
    Mode mode, StreamExecutor* executor, ::sycl::context context,
    ::sycl::device device) {
  if (executor == nullptr) {
    return absl::InvalidArgumentError(
        "SyclCommandBuffer::Create requires a non-null executor.");
  }
  return std::unique_ptr<CommandBuffer>(
      new SyclCommandBuffer(mode, executor, std::move(context),
                            std::move(device)));
}

SyclCommandBuffer::SyclCommandBuffer(Mode mode, StreamExecutor* executor,
                                     ::sycl::context context,
                                     ::sycl::device device)
    : mode_(mode),
      executor_(executor),
      context_(std::move(context)),
      device_(std::move(device)) {}

SyclCommandBuffer::~SyclCommandBuffer() = default;

absl::Status SyclCommandBuffer::CheckState(State expected,
                                           absl::string_view caller) const {
  if (state_ != expected) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "%s can be called only in %s state; current state is %s.", caller,
        StateName(expected), StateName(state_)));
  }
  return absl::OkStatus();
}

absl::StatusOr<SyclCommandBuffer::SyclCommand*>
SyclCommandBuffer::FindOwnedCommand(const Command* command,
                                    absl::string_view caller) const {
  for (const auto& owned : commands_) {
    if (owned.get() == command) {
      return owned.get();
    }
  }
  return absl::InvalidArgumentError(
      absl::StrCat(caller, " received a command not owned by this "
                           "SYCL command buffer."));
}

absl::StatusOr<std::unique_ptr<SyclCommandBuffer>>
SyclCommandBuffer::CloneForChild() const {
  auto clone = std::unique_ptr<SyclCommandBuffer>(new SyclCommandBuffer(
      mode_, executor_, context_, device_));
  clone->state_ = state_;
  clone->commands_.reserve(commands_.size());
  absl::flat_hash_map<const CommandBuffer::Command*, SyclCommand*>
      cloned_commands;
  for (const auto& command : commands_) {
    std::unique_ptr<SyclCommand> cloned_command = command->ClonePayload();
    if (cloned_command == nullptr) {
      return absl::InternalError(
          "Failed to clone a SYCL command buffer command.");
    }
    cloned_commands[command.get()] = cloned_command.get();
    clone->commands_.push_back(std::move(cloned_command));
  }
  for (int i = 0; i < commands_.size(); ++i) {
    clone->commands_[i]->dependencies.reserve(
        commands_[i]->dependencies.size());
    for (const CommandBuffer::Command* dependency :
         commands_[i]->dependencies) {
      auto it = cloned_commands.find(dependency);
      if (it == cloned_commands.end()) {
        return absl::InternalError(
            "Failed to remap SYCL command buffer child dependencies.");
      }
      clone->commands_[i]->dependencies.push_back(it->second);
    }
  }
  if (graph_artifacts_ != nullptr) {
    clone->graph_artifacts_ = std::make_unique<GraphArtifacts>();
    if (graph_artifacts_->graph != nullptr) {
      clone->graph_artifacts_->graph =
          std::make_unique<ModifiableGraph>(*graph_artifacts_->graph);
    }
    if (graph_artifacts_->executable != nullptr) {
      clone->graph_artifacts_->executable =
          std::make_unique<ExecutableGraph>(*graph_artifacts_->executable);
    }
  }
  return clone;
}

absl::StatusOr<const CommandBuffer::Command*> SyclCommandBuffer::CreateEmptyCmd(
    absl::Span<const Command* const> dependencies, StreamPriority priority) {
  RETURN_IF_ERROR(CheckState(State::kCreate, "CreateEmptyCmd"));
  commands_.push_back(std::make_unique<EmptyCommand>(dependencies));
  return commands_.back().get();
}

absl::StatusOr<const CommandBuffer::Command*> SyclCommandBuffer::CreateLaunch(
    const ThreadDim& threads, const BlockDim& blocks, const Kernel& kernel,
    const KernelArgs& args, absl::Span<const Command* const> dependencies,
    StreamPriority priority) {
  RETURN_IF_ERROR(CheckState(State::kCreate, "CreateLaunch"));
  ASSIGN_OR_RETURN(LaunchParams params,
                   SnapshotLaunchParams(threads, blocks, kernel, args));
  commands_.push_back(
      std::make_unique<LaunchCommand>(std::move(params), dependencies));
  return commands_.back().get();
}

absl::Status SyclCommandBuffer::UpdateLaunch(const Command* command,
                                             const ThreadDim& threads,
                                             const BlockDim& blocks,
                                             const Kernel& kernel,
                                             const KernelArgs& args) {
  RETURN_IF_ERROR(CheckState(State::kUpdate, "UpdateLaunch"));
  ASSIGN_OR_RETURN(SyclCommand * sycl_command,
                   FindOwnedCommand(command, "UpdateLaunch"));
  if (sycl_command->kind != SyclCommand::Kind::kLaunch) {
    return absl::InvalidArgumentError(absl::StrCat(
        "UpdateLaunch expected a launch command, got ",
        CommandKindName(sycl_command->kind), "."));
  }
  ASSIGN_OR_RETURN(LaunchArgsSnapshot snapshot,
                   SnapshotLaunchArgs(kernel, args));
  auto* launch = static_cast<LaunchCommand*>(sycl_command);
  bool can_update_node =
      graph_artifacts_ != nullptr && graph_artifacts_->executable != nullptr &&
      launch->graph_node.has_value() &&
      launch->params.kernel == snapshot.kernel &&
      launch->params.function == snapshot.function &&
      launch->params.args.size() == snapshot.args.size() &&
      launch->params.shared_mem_bytes == snapshot.shared_mem_bytes;
  if (can_update_node) {
    launch->params.threads = threads;
    launch->params.blocks = blocks;
    launch->params.args = std::move(snapshot.args);
    RefreshDynamicArgs(*launch);
    launch->node_update_pending = true;
  } else {
    LaunchParams params;
    params.threads = threads;
    params.blocks = blocks;
    params.kernel = snapshot.kernel;
    params.function = snapshot.function;
    params.kernel_name = std::string(kernel.name());
    params.args = std::move(snapshot.args);
    params.shared_mem_bytes = snapshot.shared_mem_bytes;
    launch->params = std::move(params);
    requires_graph_update_ = true;
    ++graph_update_reasons_.incompatible_launch;
  }
  return absl::OkStatus();
}

absl::StatusOr<const CommandBuffer::Command*>
SyclCommandBuffer::CreateChildCommand(
    const CommandBuffer& nested, absl::Span<const Command* const> dependencies) {
  RETURN_IF_ERROR(CheckState(State::kCreate, "CreateChildCommand"));
  const auto* nested_sycl = dynamic_cast<const SyclCommandBuffer*>(&nested);
  if (nested_sycl == nullptr) {
    return absl::InvalidArgumentError(
        "SyclCommandBuffer::CreateChildCommand requires a SYCL nested command "
        "buffer.");
  }
  if (nested_sycl->mode() != Mode::kNested) {
    return absl::InvalidArgumentError(
        "SyclCommandBuffer::CreateChildCommand requires a nested command "
        "buffer.");
  }
  ASSIGN_OR_RETURN(std::unique_ptr<SyclCommandBuffer> child,
                   nested_sycl->CloneForChild());
  commands_.push_back(
      std::make_unique<ChildCommand>(std::move(child), dependencies));
  return commands_.back().get();
}

absl::Status SyclCommandBuffer::UpdateChildCommand(
    const Command* command, const CommandBuffer& nested) {
  RETURN_IF_ERROR(CheckState(State::kUpdate, "UpdateChildCommand"));
  ASSIGN_OR_RETURN(SyclCommand * sycl_command,
                   FindOwnedCommand(command, "UpdateChildCommand"));
  if (sycl_command->kind != SyclCommand::Kind::kChild) {
    return absl::InvalidArgumentError(absl::StrCat(
        "UpdateChildCommand expected a child command, got ",
        CommandKindName(sycl_command->kind), "."));
  }
  const auto* nested_sycl = dynamic_cast<const SyclCommandBuffer*>(&nested);
  if (nested_sycl == nullptr) {
    return absl::InvalidArgumentError(
        "SyclCommandBuffer::UpdateChildCommand requires a SYCL nested command "
        "buffer.");
  }
  ASSIGN_OR_RETURN(std::unique_ptr<SyclCommandBuffer> child,
                   nested_sycl->CloneForChild());
  static_cast<ChildCommand*>(sycl_command)->child = std::move(child);
  requires_graph_update_ = true;
  ++graph_update_reasons_.child;
  return absl::OkStatus();
}

absl::StatusOr<const CommandBuffer::Command*>
SyclCommandBuffer::CreateChildCommand(
    absl::AnyInvocable<absl::Status(CommandBuffer*)> record_fn,
    absl::Span<const Command* const> dependencies) {
  RETURN_IF_ERROR(CheckState(State::kCreate, "CreateChildCommand"));
  auto child = std::unique_ptr<SyclCommandBuffer>(
      new SyclCommandBuffer(Mode::kNested, executor_, context_, device_));
  RETURN_IF_ERROR(record_fn(child.get()));
  RETURN_IF_ERROR(child->Finalize());
  commands_.push_back(
      std::make_unique<ChildCommand>(std::move(child), dependencies));
  return commands_.back().get();
}

absl::Status SyclCommandBuffer::UpdateChildCommand(
    const Command* command,
    absl::AnyInvocable<absl::Status(CommandBuffer*)> update_fn) {
  RETURN_IF_ERROR(CheckState(State::kUpdate, "UpdateChildCommand"));
  ASSIGN_OR_RETURN(SyclCommand * sycl_command,
                   FindOwnedCommand(command, "UpdateChildCommand"));
  if (sycl_command->kind != SyclCommand::Kind::kChild) {
    return absl::InvalidArgumentError(absl::StrCat(
        "UpdateChildCommand expected a child command, got ",
        CommandKindName(sycl_command->kind), "."));
  }
  auto* child = static_cast<ChildCommand*>(sycl_command)->child.get();
  RETURN_IF_ERROR(child->Update());
  RETURN_IF_ERROR(update_fn(child));
  RETURN_IF_ERROR(child->Finalize());
  requires_graph_update_ = true;
  ++graph_update_reasons_.child;
  return absl::OkStatus();
}

absl::StatusOr<const CommandBuffer::Command*> SyclCommandBuffer::CreateMemcpyD2D(
    DeviceAddressBase* dst, const DeviceAddressBase& src, uint64_t size,
    absl::Span<const Command* const> dependencies) {
  RETURN_IF_ERROR(CheckState(State::kCreate, "CreateMemcpyD2D"));
  if (dst == nullptr) {
    return absl::InvalidArgumentError(
        "SyclCommandBuffer::CreateMemcpyD2D requires a non-null destination.");
  }
  commands_.push_back(
      std::make_unique<MemcpyD2DCommand>(*dst, src, size, dependencies));
  return commands_.back().get();
}

absl::Status SyclCommandBuffer::UpdateMemcpyD2D(
    const Command* command, DeviceAddressBase* dst,
    const DeviceAddressBase& src, uint64_t size) {
  RETURN_IF_ERROR(CheckState(State::kUpdate, "UpdateMemcpyD2D"));
  if (dst == nullptr) {
    return absl::InvalidArgumentError(
        "SyclCommandBuffer::UpdateMemcpyD2D requires a non-null destination.");
  }
  ASSIGN_OR_RETURN(SyclCommand * sycl_command,
                   FindOwnedCommand(command, "UpdateMemcpyD2D"));
  if (sycl_command->kind != SyclCommand::Kind::kMemcpyD2D) {
    return absl::InvalidArgumentError(absl::StrCat(
        "UpdateMemcpyD2D expected a memcpy D2D command, got ",
        CommandKindName(sycl_command->kind), "."));
  }
  auto* memcpy = static_cast<MemcpyD2DCommand*>(sycl_command);
  memcpy->destination = *dst;
  memcpy->source = src;
  memcpy->size = size;
  requires_graph_update_ = true;
  ++graph_update_reasons_.memcpy;
  return absl::OkStatus();
}

absl::StatusOr<const CommandBuffer::Command*> SyclCommandBuffer::CreateMemset(
    DeviceAddressBase* dst, BitPattern bit_pattern, size_t num_elements,
    absl::Span<const Command* const> dependencies) {
  RETURN_IF_ERROR(CheckState(State::kCreate, "CreateMemset"));
  if (dst == nullptr) {
    return absl::InvalidArgumentError(
        "SyclCommandBuffer::CreateMemset requires a non-null destination.");
  }
  commands_.push_back(std::make_unique<MemsetCommand>(
      *dst, bit_pattern, num_elements, dependencies));
  return commands_.back().get();
}

absl::Status SyclCommandBuffer::UpdateMemset(const Command* command,
                                             DeviceAddressBase* dst,
                                             const BitPattern& bit_pattern,
                                             size_t num_elements) {
  RETURN_IF_ERROR(CheckState(State::kUpdate, "UpdateMemset"));
  if (dst == nullptr) {
    return absl::InvalidArgumentError(
        "SyclCommandBuffer::UpdateMemset requires a non-null destination.");
  }
  ASSIGN_OR_RETURN(SyclCommand * sycl_command,
                   FindOwnedCommand(command, "UpdateMemset"));
  if (sycl_command->kind != SyclCommand::Kind::kMemset) {
    return absl::InvalidArgumentError(absl::StrCat(
        "UpdateMemset expected a memset command, got ",
        CommandKindName(sycl_command->kind), "."));
  }
  auto* memset = static_cast<MemsetCommand*>(sycl_command);
  memset->destination = *dst;
  memset->bit_pattern = bit_pattern;
  memset->num_elements = num_elements;
  requires_graph_update_ = true;
  ++graph_update_reasons_.memset;
  return absl::OkStatus();
}

absl::StatusOr<const CommandBuffer::Command*>
SyclCommandBuffer::CreateDnnGraphCommand(
    dnn::DnnGraph&, Stream&, absl::Span<DeviceAddressBase> operands,
    absl::Span<const Command* const> dependencies) {
  return absl::UnimplementedError(
      "SYCL command buffers do not support DNN graph commands yet.");
}

absl::Status SyclCommandBuffer::UpdateDnnGraphCommand(
    const Command*, dnn::DnnGraph&, Stream&,
    absl::Span<DeviceAddressBase> operands) {
  return absl::UnimplementedError(
      "SYCL command buffers do not support DNN graph commands yet.");
}

absl::StatusOr<const CommandBuffer::Command*> SyclCommandBuffer::CreateCase(
    DeviceAddress<int32_t> index, std::vector<CreateCommands> create_branches,
    absl::Span<const Command* const> dependencies) {
  return absl::UnimplementedError(UnsupportedControlFlowMessage("case"));
}

absl::StatusOr<const CommandBuffer::Command*> SyclCommandBuffer::CreateCase(
    DeviceAddress<bool> index, std::vector<CreateCommands> create_branches,
    absl::Span<const Command* const> dependencies) {
  return absl::UnimplementedError(UnsupportedControlFlowMessage("case"));
}

absl::Status SyclCommandBuffer::UpdateCase(
    const Command* command, DeviceAddress<int32_t> index,
    std::vector<UpdateCommands> update_branches) {
  return absl::UnimplementedError(UnsupportedControlFlowMessage("case"));
}

absl::Status SyclCommandBuffer::UpdateCase(
    const Command* command, DeviceAddress<bool> index,
    std::vector<UpdateCommands> update_branches) {
  return absl::UnimplementedError(UnsupportedControlFlowMessage("case"));
}

absl::StatusOr<const CommandBuffer::Command*> SyclCommandBuffer::CreateWhile(
    DeviceAddress<bool> pred, CreateCommands create_cond,
    CreateCommands create_body, absl::Span<const Command* const> dependencies) {
  return absl::UnimplementedError(UnsupportedControlFlowMessage("while"));
}

absl::Status SyclCommandBuffer::UpdateWhile(
    const Command* command, DeviceAddress<bool> pred, UpdateCommands update_cond,
    UpdateCommands update_body) {
  return absl::UnimplementedError(UnsupportedControlFlowMessage("while"));
}

absl::Status SyclCommandBuffer::SetPriority(StreamPriority priority) {
  if (priority != StreamPriority::Default) {
    return absl::UnimplementedError(
        "SYCL command buffers do not support non-default stream priority.");
  }
  return absl::OkStatus();
}

absl::Status SyclCommandBuffer::RebuildGraph() {
  auto artifacts = std::make_unique<GraphArtifacts>();
  auto start = std::chrono::steady_clock::now();
  size_t leaf_count = 0;
  try {
    artifacts->graph = std::make_unique<ModifiableGraph>(context_, device_);
    ASSIGN_OR_RETURN(std::vector<GraphNode> leaves,
                     AppendCommandsToGraph(*artifacts->graph, commands_, {}));
    leaf_count = leaves.size();
    (void)leaves;
    if (artifacts->graph->get_root_nodes().empty()) {
      AddEmptyNode(*artifacts->graph, {});
    }
    auto executable = artifacts->graph->finalize(
        ::sycl::property_list{sycl_exp::property::graph::updatable{}});
    artifacts->executable =
        std::make_unique<ExecutableGraph>(std::move(executable));
  } catch (const ::sycl::exception& e) {
    return absl::InternalError(
        absl::StrCat("SYCL command graph error: ", e.what()));
  } catch (const std::exception& e) {
    return absl::InternalError(
        absl::StrCat("SYCL command graph error: ", e.what()));
  }
  VLOG(3) << "SYCL command graph finalize: commands="
          << CountCommands(commands_) << " roots="
          << artifacts->graph->get_root_nodes().size()
          << " leaves=" << leaf_count
          << " elapsed_us=" << ElapsedMicros(start);
  graph_artifacts_ = std::move(artifacts);
  return absl::OkStatus();
}

absl::Status SyclCommandBuffer::UpdateExecutableGraph() {
  if (graph_artifacts_ == nullptr || graph_artifacts_->executable == nullptr) {
    return absl::InternalError(
        "SYCL command buffer update requires an existing executable graph.");
  }

  auto start = std::chrono::steady_clock::now();
  if (requires_graph_update_) {
    absl::Status update_status = absl::OkStatus();
    try {
      auto updated_graph = std::make_unique<ModifiableGraph>(context_, device_);
      ASSIGN_OR_RETURN(
          std::vector<GraphNode> leaves,
          AppendCommandsToGraph(*updated_graph, commands_, {}));
      (void)leaves;
      if (updated_graph->get_root_nodes().empty()) {
        AddEmptyNode(*updated_graph, {});
      }
      graph_artifacts_->executable->update(*updated_graph);
      graph_artifacts_->graph = std::move(updated_graph);
    } catch (const ::sycl::exception& e) {
      update_status = absl::InternalError(
          absl::StrCat("SYCL command graph update error: ", e.what()));
    } catch (const std::exception& e) {
      update_status = absl::InternalError(
          absl::StrCat("SYCL command graph update error: ", e.what()));
    }
    if (!update_status.ok()) {
      LOG(WARNING) << update_status
                   << "; falling back to SYCL command graph re-finalization.";
      RETURN_IF_ERROR(RebuildGraph());
      VLOG(3) << "SYCL command graph update: commands="
              << CountCommands(commands_)
              << " mode=refinalize elapsed_us=" << ElapsedMicros(start)
              << " reason_memcpy=" << graph_update_reasons_.memcpy
              << " reason_memset=" << graph_update_reasons_.memset
              << " reason_child=" << graph_update_reasons_.child
              << " reason_incompatible_launch="
              << graph_update_reasons_.incompatible_launch;
    } else {
      VLOG(3) << "SYCL command graph update: commands="
              << CountCommands(commands_)
              << " mode=graph elapsed_us=" << ElapsedMicros(start)
              << " reason_memcpy=" << graph_update_reasons_.memcpy
              << " reason_memset=" << graph_update_reasons_.memset
              << " reason_child=" << graph_update_reasons_.child
              << " reason_incompatible_launch="
              << graph_update_reasons_.incompatible_launch;
    }
    requires_graph_update_ = false;
    graph_update_reasons_ = GraphUpdateReasonCounts();
    ClearPendingLaunchNodeUpdates(commands_);
    return absl::OkStatus();
  }

  std::vector<GraphNode> nodes;
  try {
    CollectPendingLaunchNodes(commands_, &nodes);
    if (!nodes.empty()) {
      graph_artifacts_->executable->update(nodes);
    }
  } catch (const ::sycl::exception& e) {
    return absl::InternalError(
        absl::StrCat("SYCL command graph node update error: ", e.what()));
  } catch (const std::exception& e) {
    return absl::InternalError(
        absl::StrCat("SYCL command graph node update error: ", e.what()));
  }
  VLOG(3) << "SYCL command graph update: commands=" << CountCommands(commands_)
          << " mode=nodes node_updates="
          << nodes.size() << " elapsed_us=" << ElapsedMicros(start);
  ClearPendingLaunchNodeUpdates(commands_);
  return absl::OkStatus();
}

absl::Status SyclCommandBuffer::Finalize() {
  if (state_ != State::kCreate && state_ != State::kUpdate) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "Finalize can be called only in create or update state; current state "
        "is %s.",
        StateName(state_)));
  }
  if (state_ == State::kUpdate) {
    RETURN_IF_ERROR(UpdateExecutableGraph());
    state_ = State::kFinalized;
    return absl::OkStatus();
  }

  if (graph_artifacts_ != nullptr && graph_artifacts_->graph != nullptr &&
      commands_.empty()) {
    auto start = std::chrono::steady_clock::now();
    try {
      if (graph_artifacts_->graph->get_root_nodes().empty()) {
        AddEmptyNode(*graph_artifacts_->graph, {});
      }
      auto executable = graph_artifacts_->graph->finalize(
          ::sycl::property_list{sycl_exp::property::graph::updatable{}});
      graph_artifacts_->executable =
          std::make_unique<ExecutableGraph>(std::move(executable));
    } catch (const ::sycl::exception& e) {
      return absl::InternalError(
          absl::StrCat("SYCL command graph finalization error: ", e.what()));
    } catch (const std::exception& e) {
      return absl::InternalError(
          absl::StrCat("SYCL command graph finalization error: ", e.what()));
    }
    VLOG(3) << "SYCL traced command graph finalize: roots="
            << graph_artifacts_->graph->get_root_nodes().size()
            << " elapsed_us=" << ElapsedMicros(start);
  } else {
    RETURN_IF_ERROR(RebuildGraph());
  }
  state_ = State::kFinalized;
  return absl::OkStatus();
}

absl::Status SyclCommandBuffer::Update() {
  RETURN_IF_ERROR(CheckState(State::kFinalized, "Update"));
  state_ = State::kUpdate;
  requires_graph_update_ = false;
  graph_update_reasons_ = GraphUpdateReasonCounts();
  ClearPendingLaunchNodeUpdates(commands_);
  return absl::OkStatus();
}

absl::Status SyclCommandBuffer::Submit(Stream* stream) {
  RETURN_IF_ERROR(CheckState(State::kFinalized, "Submit"));
  if (mode_ != Mode::kPrimary) {
    return absl::FailedPreconditionError(
        "Only primary SYCL command buffers can be submitted.");
  }
  if (graph_artifacts_ == nullptr || graph_artifacts_->executable == nullptr) {
    return absl::InternalError(
        "SYCL command buffer was finalized without an executable graph.");
  }
  auto* sycl_stream = dynamic_cast<SyclStream*>(stream);
  if (sycl_stream == nullptr) {
    return absl::InvalidArgumentError(
        "SyclCommandBuffer::Submit requires a SYCL stream.");
  }
  auto start = std::chrono::steady_clock::now();
  try {
    sycl_stream->stream_handle()->ext_oneapi_graph(
        *graph_artifacts_->executable);
  } catch (const ::sycl::exception& e) {
    return absl::InternalError(
        absl::StrCat("Failed to submit SYCL command graph: ", e.what()));
  }
  VLOG(3) << "SYCL command graph submit: commands="
          << CountCommands(commands_)
          << " elapsed_us=" << ElapsedMicros(start);
  return absl::OkStatus();
}

std::string SyclCommandBuffer::ToString() const {
  std::string dot = absl::StrFormat(
      "digraph sycl_command_buffer {\n  label=\"SYCL command buffer: %s, "
      "%s\";\n",
      ModeName(mode_), StateName(state_));
  absl::flat_hash_map<const CommandBuffer::Command*, int> command_indices;
  for (int i = 0; i < commands_.size(); ++i) {
    command_indices[commands_[i].get()] = i;
    absl::StrAppend(&dot, "  n", i, " [label=\"", commands_[i]->Label(),
                    "\"];\n");
  }
  for (int i = 0; i < commands_.size(); ++i) {
    for (const CommandBuffer::Command* dependency :
         commands_[i]->dependencies) {
      auto it = command_indices.find(dependency);
      if (it != command_indices.end()) {
        absl::StrAppend(&dot, "  n", it->second, " -> n", i, ";\n");
      }
    }
  }
  absl::StrAppend(&dot, "}\n");
  return dot;
}

absl::Status SyclCommandBuffer::Trace(
    Stream* stream, absl::AnyInvocable<absl::Status(Stream* stream)> function) {
  RETURN_IF_ERROR(CheckState(State::kCreate, "Trace"));
  if (!commands_.empty() || graph_artifacts_ != nullptr) {
    return absl::FailedPreconditionError(
        "SYCL command buffer tracing requires an empty command buffer.");
  }
  auto* sycl_stream = dynamic_cast<SyclStream*>(stream);
  if (sycl_stream == nullptr) {
    return absl::InvalidArgumentError(
        "SyclCommandBuffer::Trace requires a SYCL stream.");
  }

  auto artifacts = std::make_unique<GraphArtifacts>();
  ::sycl::queue* queue = sycl_stream->stream_handle();
  bool recording = false;
  try {
    artifacts->graph = std::make_unique<ModifiableGraph>(context_, device_);
    artifacts->graph->begin_recording(*queue);
    recording = true;
    absl::Status traced = function(stream);
    artifacts->graph->end_recording(*queue);
    recording = false;
    RETURN_IF_ERROR(traced);

    if (artifacts->graph->get_root_nodes().empty()) {
      AddEmptyNode(*artifacts->graph, {});
    }
  } catch (const ::sycl::exception& e) {
    if (recording) {
      try {
        artifacts->graph->end_recording(*queue);
      } catch (...) {
      }
    }
    return absl::InternalError(
        absl::StrCat("Failed to trace SYCL command graph: ", e.what()));
  } catch (const std::exception& e) {
    if (recording) {
      try {
        artifacts->graph->end_recording(*queue);
      } catch (...) {
      }
    }
    return absl::InternalError(
        absl::StrCat("Failed to trace SYCL command graph: ", e.what()));
  }

  graph_artifacts_ = std::move(artifacts);
  return absl::OkStatus();
}

}  // namespace stream_executor::sycl
