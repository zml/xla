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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_COMMAND_BUFFER_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_COMMAND_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "musa.h"
#include "xla/stream_executor/bit_pattern.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/gpu/gpu_command_buffer.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args_packed_vector.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::musa {

class MusaModule;

// Implements the shared GPU command-buffer contract with MUSA driver graphs.
// The qualified MUSA 4.0.1 ABI supports ordinary graph nodes and stream
// capture, but does not qualify graph conditional or child nodes.
class MusaCommandBuffer final : public gpu::GpuCommandBuffer {
 public:
  static absl::StatusOr<std::unique_ptr<MusaCommandBuffer>> Create(
      Mode mode, StreamExecutor* executor);

  ~MusaCommandBuffer() override;

  std::string ToString() const override;

 private:
  MusaCommandBuffer(Mode mode, StreamExecutor* executor, MUgraph graph)
      : GpuCommandBuffer(mode, executor), graph_(graph) {}

  absl::StatusOr<GraphNodeHandle> CreateSetCaseConditionNode(
      absl::Span<const GraphConditionalHandle> conditionals,
      DeviceAddress<uint8_t> index, bool index_is_bool, int32_t batch_offset,
      bool enable_conditional_default,
      absl::Span<const GraphNodeHandle> dependencies) override;

  absl::Status UpdateSetCaseConditionNode(
      GraphNodeHandle handle,
      absl::Span<const GraphConditionalHandle> conditionals,
      DeviceAddress<uint8_t> index, bool index_is_bool, int32_t batch_offset,
      bool enable_conditional_default) override;

  absl::StatusOr<GraphNodeHandle> CreateSetWhileConditionNode(
      GraphConditionalHandle conditional, DeviceAddress<bool> predicate,
      absl::Span<const GraphNodeHandle> dependencies) override;

  absl::Status UpdateSetWhileConditionNode(
      GraphNodeHandle handle, GraphConditionalHandle conditional,
      DeviceAddress<bool> predicate) override;

  absl::StatusOr<GraphConditionalNodeHandle> CreateConditionalNode(
      absl::Span<const GraphNodeHandle> dependencies,
      GraphConditionalHandle conditional, ConditionType type) override;

  absl::StatusOr<GraphNodeHandle> CreateMemsetNode(
      absl::Span<const GraphNodeHandle> dependencies,
      DeviceAddressBase destination, BitPattern bit_pattern,
      size_t num_elements) override;

  absl::Status UpdateMemsetNode(GraphNodeHandle node_handle,
                                DeviceAddressBase destination,
                                BitPattern bit_pattern,
                                size_t num_elements) override;

  absl::StatusOr<GraphNodeHandle> CreateMemcpyD2DNode(
      absl::Span<const GraphNodeHandle> dependencies,
      DeviceAddressBase destination, DeviceAddressBase source,
      uint64_t size) override;

  absl::Status UpdateMemcpyD2DNode(GraphNodeHandle node_handle,
                                   DeviceAddressBase destination,
                                   DeviceAddressBase source,
                                   uint64_t size) override;

  absl::Status PopulateDnnGraphNode(
      dnn::DnnGraph&, Stream&, absl::Span<DeviceAddressBase> operands) override;

  absl::Status UpdateDnnGraphNode(dnn::DnnGraph&, Stream&,
                                  absl::Span<DeviceAddressBase> operands,
                                  GraphNodeHandle node_handle) override;

  absl::StatusOr<GraphNodeHandle> CreateClonedChildNode(
      absl::Span<const GraphNodeHandle> dependencies,
      const CommandBuffer& nested) override;

  absl::StatusOr<GraphNodeHandle> CreateMovedChildNode(
      absl::Span<const GraphNodeHandle> dependencies,
      CommandBuffer* nested) override;

  absl::Status UpdateClonedChildNode(GraphNodeHandle node_handle,
                                     const CommandBuffer& nested) override;

  absl::StatusOr<GraphNodeHandle> CreateKernelNode(
      absl::Span<const GraphNodeHandle> dependencies, StreamPriority priority,
      const ThreadDim& threads, const BlockDim& blocks, const Kernel& kernel,
      const KernelArgsPackedArrayBase& args) override;

  absl::Status UpdateKernelNode(GraphNodeHandle node_handle,
                                const ThreadDim& threads,
                                const BlockDim& blocks, const Kernel& kernel,
                                const KernelArgsPackedArrayBase& args) override;

  absl::StatusOr<GraphNodeHandle> CreateEmptyNode(
      absl::Span<const GraphNodeHandle> dependencies) override;

  absl::Status Trace(
      Stream* stream,
      absl::AnyInvocable<absl::Status(Stream* stream)> function) override;

  absl::Status LaunchGraph(Stream* stream) override;
  absl::StatusOr<size_t> GetNodeCount() const override;

  absl::Status SetPriority(StreamPriority priority) override {
    // MUSA 4.0.1 has no per-node graph priority API.
    return absl::OkStatus();
  }

  absl::Status PrepareFinalization() override;
  absl::StatusOr<GraphConditionalHandle> CreateConditionalHandle() override;
  absl::Status WriteGraphToDotFile(absl::string_view path) override;
  absl::Status InstantiateGraph() override;
  absl::Status CheckCanBeUpdated() override;

  void RetainModule(const std::shared_ptr<MusaModule>& module);

  static_assert(std::is_pointer_v<MUgraph>, "MUgraph must be a pointer");
  static_assert(std::is_pointer_v<MUgraphExec>,
                "MUgraphExec must be a pointer");

  MUgraph graph_ = nullptr;
  MUgraphExec executable_ = nullptr;
  MUstream last_submission_stream_ = nullptr;
  std::unordered_map<MUgraphNode, std::shared_ptr<KernelArgsPackedVector>>
      graph_kernel_arguments_;
  std::vector<std::shared_ptr<KernelArgsPackedVector>>
      captured_kernel_arguments_;
  std::vector<std::shared_ptr<MusaModule>> retained_modules_;
};

}  // namespace stream_executor::musa

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_COMMAND_BUFFER_H_
