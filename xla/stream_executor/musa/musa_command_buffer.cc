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

#include "xla/stream_executor/musa/musa_command_buffer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "musa.h"
#include "xla/stream_executor/bit_pattern.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_command_buffer.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/musa/musa_driver.h"
#include "xla/stream_executor/musa/musa_graph_arguments.h"
#include "xla/stream_executor/musa/musa_kernel.h"
#include "xla/stream_executor/musa/musa_module.h"
#include "xla/stream_executor/musa/musa_stream.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::musa {
namespace {

using GraphNodeHandle = gpu::GpuCommandBuffer::GraphNodeHandle;

MUgraphNode ToMusaGraphHandle(GraphNodeHandle handle) {
  return absl::bit_cast<MUgraphNode>(handle);
}

GraphNodeHandle FromMusaGraphHandle(MUgraphNode handle) {
  return absl::bit_cast<GraphNodeHandle>(handle);
}

std::vector<MUgraphNode> ToMusaGraphHandles(
    absl::Span<const GraphNodeHandle> handles) {
  std::vector<MUgraphNode> native;
  native.reserve(handles.size());
  for (GraphNodeHandle handle : handles) {
    native.push_back(ToMusaGraphHandle(handle));
  }
  return native;
}

absl::Status ValidateDimensions(const ThreadDim& threads,
                                const BlockDim& blocks,
                                uint64_t shared_memory_bytes) {
  constexpr uint64_t kMaxUnsigned = std::numeric_limits<unsigned int>::max();
  if (threads.x == 0 || threads.y == 0 || threads.z == 0 || blocks.x == 0 ||
      blocks.y == 0 || blocks.z == 0) {
    return absl::InvalidArgumentError(
        "MUSA graph launch dimensions must be nonzero");
  }
  if (threads.x > kMaxUnsigned || threads.y > kMaxUnsigned ||
      threads.z > kMaxUnsigned || blocks.x > kMaxUnsigned ||
      blocks.y > kMaxUnsigned || blocks.z > kMaxUnsigned ||
      shared_memory_bytes > kMaxUnsigned) {
    return absl::InvalidArgumentError(
        "MUSA graph launch dimensions exceed the runtime ABI");
  }
  return absl::OkStatus();
}

MUSA_KERNEL_NODE_PARAMS MakeKernelParams(
    const ThreadDim& threads, const BlockDim& blocks, const MusaKernel& kernel,
    const KernelArgsPackedArrayBase& args) {
  MUSA_KERNEL_NODE_PARAMS params = {};
  params.func = kernel.function();
  params.gridDimX = static_cast<unsigned int>(blocks.x);
  params.gridDimY = static_cast<unsigned int>(blocks.y);
  params.gridDimZ = static_cast<unsigned int>(blocks.z);
  params.blockDimX = static_cast<unsigned int>(threads.x);
  params.blockDimY = static_cast<unsigned int>(threads.y);
  params.blockDimZ = static_cast<unsigned int>(threads.z);
  params.sharedMemBytes =
      static_cast<unsigned int>(args.number_of_shared_bytes());
  params.kernelParams = const_cast<void**>(args.argument_addresses().data());
  params.extra = nullptr;
  return params;
}

}  // namespace

absl::StatusOr<std::unique_ptr<MusaCommandBuffer>> MusaCommandBuffer::Create(
    Mode mode, StreamExecutor* executor) {
  if (executor == nullptr) {
    return absl::InvalidArgumentError(
        "MUSA command buffer requires a StreamExecutor");
  }
  if (!MusaDriver::Instance().GraphsAvailable()) {
    return absl::UnimplementedError(
        "The loaded MUSA driver does not provide the complete graph ABI");
  }
  ASSIGN_OR_RETURN(MUgraph graph, MusaDriver::Instance().GraphCreate());
  return std::unique_ptr<MusaCommandBuffer>(
      new MusaCommandBuffer(mode, executor, graph));
}

absl::StatusOr<GraphNodeHandle> MusaCommandBuffer::CreateSetCaseConditionNode(
    absl::Span<const GraphConditionalHandle> conditionals,
    DeviceAddress<uint8_t> index, bool index_is_bool, int32_t batch_offset,
    bool enable_conditional_default,
    absl::Span<const GraphNodeHandle> dependencies) {
  return absl::UnimplementedError(
      "MUSA 4.0.1 graphs do not support conditional nodes");
}

absl::Status MusaCommandBuffer::UpdateSetCaseConditionNode(
    GraphNodeHandle handle,
    absl::Span<const GraphConditionalHandle> conditionals,
    DeviceAddress<uint8_t> index, bool index_is_bool, int32_t batch_offset,
    bool enable_conditional_default) {
  return absl::UnimplementedError(
      "MUSA 4.0.1 graphs do not support conditional nodes");
}

absl::StatusOr<GraphNodeHandle> MusaCommandBuffer::CreateSetWhileConditionNode(
    GraphConditionalHandle conditional, DeviceAddress<bool> predicate,
    absl::Span<const GraphNodeHandle> dependencies) {
  return absl::UnimplementedError(
      "MUSA 4.0.1 graphs do not support conditional nodes");
}

absl::Status MusaCommandBuffer::UpdateSetWhileConditionNode(
    GraphNodeHandle handle, GraphConditionalHandle conditional,
    DeviceAddress<bool> predicate) {
  return absl::UnimplementedError(
      "MUSA 4.0.1 graphs do not support conditional nodes");
}

absl::StatusOr<gpu::GpuCommandBuffer::GraphConditionalNodeHandle>
MusaCommandBuffer::CreateConditionalNode(
    absl::Span<const GraphNodeHandle> dependencies,
    GraphConditionalHandle conditional, ConditionType type) {
  return absl::UnimplementedError(
      "MUSA 4.0.1 graphs do not support conditional nodes");
}

absl::StatusOr<GraphNodeHandle> MusaCommandBuffer::CreateMemsetNode(
    absl::Span<const GraphNodeHandle> dependencies,
    DeviceAddressBase destination, BitPattern bit_pattern,
    size_t num_elements) {
  MUSA_MEMSET_NODE_PARAMS params = {};
  params.dst = absl::bit_cast<MUdeviceptr>(destination.opaque());
  params.pitch = 0;
  params.value = bit_pattern.GetPatternBroadcastedToUint32();
  params.elementSize = bit_pattern.GetElementSize();
  params.width = num_elements;
  params.height = 1;
  std::vector<MUgraphNode> native = ToMusaGraphHandles(dependencies);
  ASSIGN_OR_RETURN(MUgraphNode node, MusaDriver::Instance().GraphAddMemsetNode(
                                         graph_, native, params));
  return FromMusaGraphHandle(node);
}

absl::Status MusaCommandBuffer::UpdateMemsetNode(GraphNodeHandle node_handle,
                                                 DeviceAddressBase destination,
                                                 BitPattern bit_pattern,
                                                 size_t num_elements) {
  MUSA_MEMSET_NODE_PARAMS params = {};
  params.dst = absl::bit_cast<MUdeviceptr>(destination.opaque());
  params.pitch = 0;
  params.value = bit_pattern.GetPatternBroadcastedToUint32();
  params.elementSize = bit_pattern.GetElementSize();
  params.width = num_elements;
  params.height = 1;
  return MusaDriver::Instance().GraphMemsetNodeSetParams(
      ToMusaGraphHandle(node_handle), params);
}

absl::StatusOr<GraphNodeHandle> MusaCommandBuffer::CreateMemcpyD2DNode(
    absl::Span<const GraphNodeHandle> dependencies,
    DeviceAddressBase destination, DeviceAddressBase source, uint64_t size) {
  if (size > std::numeric_limits<size_t>::max()) {
    return absl::InvalidArgumentError(
        "MUSA graph memcpy size exceeds the runtime ABI");
  }
  std::vector<MUgraphNode> native = ToMusaGraphHandles(dependencies);
  ASSIGN_OR_RETURN(
      MUgraphNode node,
      MusaDriver::Instance().GraphAddMemcpyD2DNode(
          graph_, native, absl::bit_cast<MUdeviceptr>(destination.opaque()),
          absl::bit_cast<MUdeviceptr>(source.opaque()),
          static_cast<size_t>(size)));
  return FromMusaGraphHandle(node);
}

absl::Status MusaCommandBuffer::UpdateMemcpyD2DNode(
    GraphNodeHandle node_handle, DeviceAddressBase destination,
    DeviceAddressBase source, uint64_t size) {
  if (size > std::numeric_limits<size_t>::max()) {
    return absl::InvalidArgumentError(
        "MUSA graph memcpy size exceeds the runtime ABI");
  }
  return MusaDriver::Instance().GraphMemcpyD2DNodeSetParams(
      ToMusaGraphHandle(node_handle),
      absl::bit_cast<MUdeviceptr>(destination.opaque()),
      absl::bit_cast<MUdeviceptr>(source.opaque()), static_cast<size_t>(size));
}

absl::Status MusaCommandBuffer::PopulateDnnGraphNode(
    dnn::DnnGraph&, Stream&, absl::Span<DeviceAddressBase> operands) {
  return absl::UnimplementedError(
      "muDNN graph capture is not qualified for MUSA command buffers");
}

absl::Status MusaCommandBuffer::UpdateDnnGraphNode(
    dnn::DnnGraph&, Stream&, absl::Span<DeviceAddressBase> operands,
    GraphNodeHandle node_handle) {
  return absl::UnimplementedError(
      "muDNN graph capture is not qualified for MUSA command buffers");
}

absl::StatusOr<GraphNodeHandle> MusaCommandBuffer::CreateClonedChildNode(
    absl::Span<const GraphNodeHandle> dependencies,
    const CommandBuffer& nested) {
  return absl::UnimplementedError(
      "MUSA 4.0.1 child graph nodes are not qualified on S80");
}

absl::StatusOr<GraphNodeHandle> MusaCommandBuffer::CreateMovedChildNode(
    absl::Span<const GraphNodeHandle> dependencies, CommandBuffer* nested) {
  return absl::UnimplementedError(
      "MUSA 4.0.1 does not expose mutable child graph nodes");
}

absl::Status MusaCommandBuffer::UpdateClonedChildNode(
    GraphNodeHandle node_handle, const CommandBuffer& nested) {
  return absl::UnimplementedError(
      "MUSA 4.0.1 does not expose child graph node updates");
}

absl::StatusOr<GraphNodeHandle> MusaCommandBuffer::CreateKernelNode(
    absl::Span<const GraphNodeHandle> dependencies, StreamPriority priority,
    const ThreadDim& threads, const BlockDim& blocks, const Kernel& kernel,
    const KernelArgsPackedArrayBase& args) {
  const auto* musa_kernel = dynamic_cast<const MusaKernel*>(&kernel);
  if (musa_kernel == nullptr) {
    return absl::InvalidArgumentError(
        "MUSA graph launch requires a MusaKernel");
  }
  RETURN_IF_ERROR(
      ValidateDimensions(threads, blocks, args.number_of_shared_bytes()));

  std::unique_ptr<KernelArgsPackedArrayBase> repacked;
  const KernelArgsPackedArrayBase* packed = &args;
  if (kernel.args_packing()) {
    ASSIGN_OR_RETURN(repacked, kernel.args_packing()(kernel, args));
    packed = repacked.get();
    RETURN_IF_ERROR(
        ValidateDimensions(threads, blocks, packed->number_of_shared_bytes()));
  }

  ASSIGN_OR_RETURN(std::unique_ptr<KernelArgsPackedVector> cloned_arguments,
                   CloneMusaGraphKernelArguments(*packed));
  std::shared_ptr<KernelArgsPackedVector> graph_arguments(
      std::move(cloned_arguments));
  MUSA_KERNEL_NODE_PARAMS params =
      MakeKernelParams(threads, blocks, *musa_kernel, *graph_arguments);
  std::vector<MUgraphNode> native = ToMusaGraphHandles(dependencies);
  ASSIGN_OR_RETURN(MUgraphNode node, MusaDriver::Instance().GraphAddKernelNode(
                                         graph_, native, params));
  graph_kernel_arguments_.insert_or_assign(node, std::move(graph_arguments));
  RetainModule(musa_kernel->module());
  return FromMusaGraphHandle(node);
}

absl::Status MusaCommandBuffer::UpdateKernelNode(
    GraphNodeHandle node_handle, const ThreadDim& threads,
    const BlockDim& blocks, const Kernel& kernel,
    const KernelArgsPackedArrayBase& args) {
  const auto* musa_kernel = dynamic_cast<const MusaKernel*>(&kernel);
  if (musa_kernel == nullptr) {
    return absl::InvalidArgumentError(
        "MUSA graph update requires a MusaKernel");
  }
  RETURN_IF_ERROR(
      ValidateDimensions(threads, blocks, args.number_of_shared_bytes()));

  std::unique_ptr<KernelArgsPackedArrayBase> repacked;
  const KernelArgsPackedArrayBase* packed = &args;
  if (kernel.args_packing()) {
    ASSIGN_OR_RETURN(repacked, kernel.args_packing()(kernel, args));
    packed = repacked.get();
    RETURN_IF_ERROR(
        ValidateDimensions(threads, blocks, packed->number_of_shared_bytes()));
  }

  ASSIGN_OR_RETURN(std::unique_ptr<KernelArgsPackedVector> cloned_arguments,
                   CloneMusaGraphKernelArguments(*packed));
  std::shared_ptr<KernelArgsPackedVector> graph_arguments(
      std::move(cloned_arguments));
  MUSA_KERNEL_NODE_PARAMS params =
      MakeKernelParams(threads, blocks, *musa_kernel, *graph_arguments);
  RETURN_IF_ERROR(MusaDriver::Instance().GraphKernelNodeSetParams(
      ToMusaGraphHandle(node_handle), params));
  graph_kernel_arguments_.insert_or_assign(ToMusaGraphHandle(node_handle),
                                           std::move(graph_arguments));
  RetainModule(musa_kernel->module());
  return absl::OkStatus();
}

absl::StatusOr<GraphNodeHandle> MusaCommandBuffer::CreateEmptyNode(
    absl::Span<const GraphNodeHandle> dependencies) {
  std::vector<MUgraphNode> native = ToMusaGraphHandles(dependencies);
  ASSIGN_OR_RETURN(MUgraphNode node,
                   MusaDriver::Instance().GraphAddEmptyNode(graph_, native));
  return FromMusaGraphHandle(node);
}

absl::Status MusaCommandBuffer::Trace(
    Stream* stream, absl::AnyInvocable<absl::Status(Stream* stream)> function) {
  RETURN_IF_ERROR(CheckNotFinalized());
  if (stream == nullptr) {
    return absl::InvalidArgumentError(
        "MUSA graph capture requires a non-null stream");
  }
  auto* musa_stream = dynamic_cast<MusaStream*>(stream);
  if (musa_stream == nullptr) {
    return absl::InvalidArgumentError(
        "MUSA graph capture requires a MUSA stream");
  }
  ASSIGN_OR_RETURN(size_t count, GetNodeCount());
  if (count != 0) {
    return absl::FailedPreconditionError(
        "MUSA stream capture requires an empty command buffer");
  }

  MUstream stream_handle =
      reinterpret_cast<MUstream>(stream->platform_specific_handle().stream);
  RETURN_IF_ERROR(musa_stream->BeginGraphCaptureModuleTracking());
  absl::Status begin = MusaDriver::Instance().StreamBeginCapture(stream_handle);
  if (!begin.ok()) {
    musa_stream->AbortGraphCaptureModuleTracking();
    return begin;
  }
  absl::Status traced = function(stream);
  absl::StatusOr<MUgraph> captured =
      MusaDriver::Instance().StreamEndCapture(stream_handle);
  absl::StatusOr<MusaGraphCaptureResources> captured_resources =
      musa_stream->EndGraphCaptureModuleTracking();
  if (!captured.ok()) return captured.status();
  if (!captured_resources.ok()) {
    absl::Status destroy = MusaDriver::Instance().GraphDestroy(*captured);
    if (!destroy.ok()) {
      LOG(ERROR) << "Failed to destroy untracked MUSA capture: " << destroy;
    }
    return captured_resources.status();
  }
  if (!traced.ok()) {
    absl::Status destroy = MusaDriver::Instance().GraphDestroy(*captured);
    if (!destroy.ok()) {
      LOG(ERROR) << "Failed to destroy rejected MUSA capture: " << destroy;
    }
    return absl::Status(
        traced.code(),
        absl::StrCat("Failed to capture MUSA graph: ", traced.message()));
  }

  absl::Status destroyed = MusaDriver::Instance().GraphDestroy(graph_);
  if (!destroyed.ok()) {
    absl::Status cleanup = MusaDriver::Instance().GraphDestroy(*captured);
    if (!cleanup.ok()) {
      LOG(ERROR) << "Failed to destroy replacement MUSA capture: " << cleanup;
    }
    return destroyed;
  }
  graph_ = *captured;
  for (const std::shared_ptr<MusaModule>& module :
       captured_resources->modules) {
    RetainModule(module);
  }
  captured_kernel_arguments_ = std::move(captured_resources->kernel_arguments);
  ASSIGN_OR_RETURN(size_t roots,
                   MusaDriver::Instance().GraphRootNodeCount(graph_));
  if (roots == 0) {
    ASSIGN_OR_RETURN(GraphNodeHandle empty, CreateEmptyNode({}));
    (void)empty;
  }
  return absl::OkStatus();
}

absl::Status MusaCommandBuffer::LaunchGraph(Stream* stream) {
  if (stream == nullptr) {
    return absl::InvalidArgumentError(
        "MUSA graph launch requires a non-null stream");
  }
  if (executable_ == nullptr) {
    return absl::FailedPreconditionError(
        "MUSA graph must be finalized before launch");
  }
  MUstream stream_handle =
      reinterpret_cast<MUstream>(stream->platform_specific_handle().stream);
  absl::Status launched =
      MusaDriver::Instance().GraphLaunch(executable_, stream_handle);
  if (launched.ok()) last_submission_stream_ = stream_handle;
  return launched;
}

absl::StatusOr<size_t> MusaCommandBuffer::GetNodeCount() const {
  return MusaDriver::Instance().GraphNodeCount(graph_);
}

absl::Status MusaCommandBuffer::PrepareFinalization() {
  ASSIGN_OR_RETURN(size_t count, GetNodeCount());
  if (count == 0) {
    ASSIGN_OR_RETURN(GraphNodeHandle empty, CreateEmptyNode({}));
    (void)empty;
  }
  if (mode() == Mode::kPrimary && state() == State::kUpdate) {
    ASSIGN_OR_RETURN(MUgraphExec replacement,
                     MusaDriver::Instance().GraphInstantiate(graph_));
    absl::Status destroyed =
        MusaDriver::Instance().GraphExecDestroy(executable_);
    if (!destroyed.ok()) {
      absl::Status cleanup =
          MusaDriver::Instance().GraphExecDestroy(replacement);
      if (!cleanup.ok()) {
        LOG(ERROR) << "Failed to destroy replacement MUSA executable graph: "
                   << cleanup;
      }
      return destroyed;
    }
    executable_ = replacement;
  }
  return absl::OkStatus();
}

absl::StatusOr<gpu::GpuCommandBuffer::GraphConditionalHandle>
MusaCommandBuffer::CreateConditionalHandle() {
  return absl::UnimplementedError(
      "MUSA 4.0.1 graphs do not support conditional nodes");
}

absl::Status MusaCommandBuffer::WriteGraphToDotFile(absl::string_view path) {
  return absl::UnimplementedError(
      "MUSA 4.0.1 does not export graph DOT serialization");
}

absl::Status MusaCommandBuffer::InstantiateGraph() {
  ASSIGN_OR_RETURN(executable_,
                   MusaDriver::Instance().GraphInstantiate(graph_));
  return absl::OkStatus();
}

absl::Status MusaCommandBuffer::CheckCanBeUpdated() {
  if (mode() != Mode::kPrimary) {
    return absl::UnimplementedError(
        "MUSA 4.0.1 does not expose mutable child graph nodes");
  }
  if (executable_ == nullptr) {
    return absl::FailedPreconditionError(
        "MUSA command buffer must have an executable graph before update");
  }
  // Shared command-buffer caching can begin the next update before an earlier
  // asynchronous launch completes. MUSA source-node parameters and their
  // replacement executable must not be changed while that launch is in flight,
  // so establish an explicit backend lifetime boundary first.
  if (last_submission_stream_ != nullptr) {
    RETURN_IF_ERROR(
        MusaDriver::Instance().StreamSynchronize(last_submission_stream_));
    last_submission_stream_ = nullptr;
  }
  return absl::OkStatus();
}

void MusaCommandBuffer::RetainModule(
    const std::shared_ptr<MusaModule>& module) {
  if (module == nullptr) return;
  auto duplicate =
      std::find_if(retained_modules_.begin(), retained_modules_.end(),
                   [&module](const std::shared_ptr<MusaModule>& retained) {
                     return retained.get() == module.get();
                   });
  if (duplicate == retained_modules_.end()) {
    retained_modules_.push_back(module);
  }
}

MusaCommandBuffer::~MusaCommandBuffer() {
  if (executable_ != nullptr) {
    int64_t remaining = NotifyExecDestroyed();
    absl::Status status = MusaDriver::Instance().GraphExecDestroy(executable_);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to destroy MUSA executable graph: " << status;
    }
    VLOG(5) << "Destroyed MUSA executable graph; remaining=" << remaining;
  }
  if (graph_ != nullptr) {
    absl::Status status = MusaDriver::Instance().GraphDestroy(graph_);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to destroy MUSA graph: " << status;
    }
  }
}

std::string MusaCommandBuffer::ToString() const {
  absl::StatusOr<size_t> count = GetNodeCount();
  if (!count.ok()) {
    return absl::StrCat("MUSA graph node query failed: ", count.status());
  }
  return absl::StrCat("MUSA graph with ", *count, " nodes");
}

}  // namespace stream_executor::musa
