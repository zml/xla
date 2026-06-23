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

#ifndef XLA_STREAM_EXECUTOR_SYCL_SYCL_COMMAND_BUFFER_H_
#define XLA_STREAM_EXECUTOR_SYCL_SYCL_COMMAND_BUFFER_H_

#include <memory>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>  // NOLINT

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/bit_pattern.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"

namespace stream_executor::sycl {

class SyclCommandBuffer final : public CommandBuffer {
 public:
  struct GraphArtifacts;
  struct SyclCommand;

  static absl::StatusOr<std::unique_ptr<CommandBuffer>> Create(
      Mode mode, StreamExecutor* executor, ::sycl::context context,
      ::sycl::device device);

  ~SyclCommandBuffer() override;

  absl::StatusOr<const Command*> CreateEmptyCmd(
      absl::Span<const Command* const> dependencies,
      StreamPriority priority = StreamPriority::Default) override;

  absl::StatusOr<const Command*> CreateLaunch(
      const ThreadDim& threads, const BlockDim& blocks, const Kernel& kernel,
      const KernelArgs& args, absl::Span<const Command* const> dependencies,
      StreamPriority priority = StreamPriority::Default) override;

  absl::Status UpdateLaunch(const Command* command, const ThreadDim& threads,
                            const BlockDim& blocks, const Kernel& kernel,
                            const KernelArgs& args) override;

  absl::StatusOr<const Command*> CreateChildCommand(
      const CommandBuffer& nested,
      absl::Span<const Command* const> dependencies) override;

  absl::Status UpdateChildCommand(const Command* command,
                                  const CommandBuffer& nested) override;

  absl::StatusOr<const Command*> CreateChildCommand(
      absl::AnyInvocable<absl::Status(CommandBuffer*)> record_fn,
      absl::Span<const Command* const> dependencies) override;

  absl::Status UpdateChildCommand(
      const Command* command,
      absl::AnyInvocable<absl::Status(CommandBuffer*)> update_fn) override;

  absl::StatusOr<const Command*> CreateMemcpyD2D(
      DeviceAddressBase* dst, const DeviceAddressBase& src, uint64_t size,
      absl::Span<const Command* const> dependencies) override;

  absl::Status UpdateMemcpyD2D(const Command* command, DeviceAddressBase* dst,
                               const DeviceAddressBase& src,
                               uint64_t size) override;

  absl::StatusOr<const Command*> CreateMemset(
      DeviceAddressBase* dst, BitPattern bit_pattern, size_t num_elements,
      absl::Span<const Command* const> dependencies) override;

  absl::Status UpdateMemset(const Command* command, DeviceAddressBase* dst,
                            const BitPattern& bit_pattern,
                            size_t num_elements) override;

  absl::StatusOr<const Command*> CreateDnnGraphCommand(
      dnn::DnnGraph&, Stream&, absl::Span<DeviceAddressBase> operands,
      absl::Span<const Command* const> dependencies) override;

  absl::Status UpdateDnnGraphCommand(
      const Command*, dnn::DnnGraph&, Stream&,
      absl::Span<DeviceAddressBase> operands) override;

  absl::StatusOr<const Command*> CreateCase(
      DeviceAddress<int32_t> index, std::vector<CreateCommands> create_branches,
      absl::Span<const Command* const> dependencies) override;

  absl::StatusOr<const Command*> CreateCase(
      DeviceAddress<bool> index, std::vector<CreateCommands> create_branches,
      absl::Span<const Command* const> dependencies) override;

  absl::Status UpdateCase(const Command* command, DeviceAddress<int32_t> index,
                          std::vector<UpdateCommands> update_branches) override;

  absl::Status UpdateCase(const Command* command, DeviceAddress<bool> index,
                          std::vector<UpdateCommands> update_branches) override;

  absl::StatusOr<const Command*> CreateWhile(
      DeviceAddress<bool> pred, CreateCommands create_cond,
      CreateCommands create_body,
      absl::Span<const Command* const> dependencies) override;

  absl::Status UpdateWhile(const Command* command, DeviceAddress<bool> pred,
                           UpdateCommands update_cond,
                           UpdateCommands update_body) override;

  absl::Status SetPriority(StreamPriority priority) override;
  absl::Status Submit(Stream* stream) override;

  absl::Status Finalize() override;
  absl::Status Update() override;

  Mode mode() const override { return mode_; }
  State state() const override { return state_; }

  std::string ToString() const override;

  absl::StatusOr<std::unique_ptr<SyclCommandBuffer>> CloneForChild() const;
  const std::vector<std::unique_ptr<SyclCommand>>& commands() const {
    return commands_;
  }
  const GraphArtifacts* graph_artifacts() const {
    return graph_artifacts_.get();
  }

 private:
  SyclCommandBuffer(Mode mode, StreamExecutor* executor,
                    ::sycl::context context, ::sycl::device device);

  absl::Status Trace(Stream* stream,
                     absl::AnyInvocable<absl::Status(Stream* stream)> function)
      override;

  absl::Status CheckState(State expected, absl::string_view caller) const;
  absl::StatusOr<SyclCommand*> FindOwnedCommand(
      const Command* command, absl::string_view caller) const;
  absl::Status RebuildGraph();
  absl::Status UpdateExecutableGraph();

  struct GraphUpdateReasonCounts {
    int64_t memcpy = 0;
    int64_t memset = 0;
    int64_t child = 0;
    int64_t incompatible_launch = 0;
  };

  Mode mode_;
  State state_ = State::kCreate;
  StreamExecutor* executor_;
  ::sycl::context context_;
  ::sycl::device device_;

  std::vector<std::unique_ptr<SyclCommand>> commands_;
  std::unique_ptr<GraphArtifacts> graph_artifacts_;
  bool requires_graph_update_ = false;
  GraphUpdateReasonCounts graph_update_reasons_;
};

}  // namespace stream_executor::sycl

#endif  // XLA_STREAM_EXECUTOR_SYCL_SYCL_COMMAND_BUFFER_H_
