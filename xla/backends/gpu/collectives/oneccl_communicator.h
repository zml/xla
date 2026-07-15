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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_ONECCL_COMMUNICATOR_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_ONECCL_COMMUNICATOR_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "oneapi/ccl.hpp"
#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/core/collectives/registered_memory.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/future.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

class OnecclCommunicator final : public GpuCommunicator {
 public:
  static absl::StatusOr<std::unique_ptr<OnecclCommunicator>> Create(
      se::StreamExecutor* stream_executor, ccl::communicator comm,
      std::shared_ptr<ccl::kvs_interface> kvs,
      std::shared_ptr<CancellationToken> cancel,
      tsl::Env& env = *tsl::Env::Default());

  ~OnecclCommunicator() final;

  OnecclCommunicator(const OnecclCommunicator&) = delete;
  OnecclCommunicator(OnecclCommunicator&&) = delete;
  OnecclCommunicator& operator=(const OnecclCommunicator&) = delete;
  OnecclCommunicator& operator=(OnecclCommunicator&&) = delete;

  absl::Status Abort() final;
  absl::Status HealthCheck() const final;
  absl::Status Barrier(const Executor& executor) final;
  absl::StatusOr<size_t> NumRanks() const final;
  absl::StatusOr<size_t> CurrentRank() final;

  PlatformCommunicatorHandle platform_comm() const final {
    return PlatformCommunicatorHandle{comm_.get()};
  }

  bool SupportsDeviceComm() const final { return false; }

  // oneCCL's grouped SYCL point-to-point path stores a single receive-buffer
  // discovery entry per source/destination rank pair. Multiple
  // CollectivePermute buffers reuse that pair and cannot safely share a group.
  bool SupportsGroupedCollectivePermute() const final { return false; }

  absl::StatusOr<std::unique_ptr<GpuDeviceCommunicator>> CreateDeviceComm(
      const GpuDeviceCommunicator::Requirements& requirements) final;

  absl::StatusOr<std::unique_ptr<RegisteredMemory>> CreateRegisteredMemory(
      se::DeviceAddressBase addr) final;

  absl::StatusOr<std::unique_ptr<SymmetricMemory>> CreateSymmetricMemory(
      se::DeviceAddressBase addr) final;

  Future<> GroupExecute(absl::AnyInvocable<absl::Status() &&> group) final;
  Future<> GroupExecuteCounted(absl::AnyInvocable<absl::Status() &&> group,
                               int64_t num_collectives) final;

  Future<> AllReduce(se::DeviceAddressBase send_buffer,
                     se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
                     size_t count, ReductionKind reduction_kind,
                     const Executor& executor) final;

  Future<> Broadcast(se::DeviceAddressBase send_buffer,
                     se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
                     size_t count, RankId root, const Executor& executor) final;

  Future<> ReduceScatter(se::DeviceAddressBase send_buffer,
                         se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
                         size_t count, ReductionKind reduction_kind,
                         const Executor& executor) final;

  Future<> AllGather(se::DeviceAddressBase send_buffer,
                     se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
                     size_t count, const Executor& executor) final;

  Future<> AllToAll(absl::InlinedVector<se::DeviceAddressBase, 4> send_buffers,
                    absl::InlinedVector<se::DeviceAddressBase, 4> recv_buffers,
                    PrimitiveType dtype, size_t count,
                    const Executor& executor) final;

  Future<> CollectivePermute(se::DeviceAddressBase send_buffer,
                             se::DeviceAddressBase recv_buffer,
                             PrimitiveType dtype, size_t count,
                             std::optional<RankId> source_rank,
                             absl::Span<const RankId> target_ranks,
                             const Executor& executor) final;

  Future<> Send(se::DeviceAddressBase send_buffer, PrimitiveType dtype,
                size_t count, RankId peer, const Executor& executor) final;

  Future<> Recv(se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
                size_t count, RankId peer, const Executor& executor) final;

  Future<> Put(se::DeviceAddressBase send_buffer, SymmetricMemory* recv_buffer,
               size_t offset, size_t count, RankId peer,
               const Executor& executor) final;

  Future<> Signal(RankId peer, const SignalDesc& signal_desc,
                  const Executor& executor) final;

  Future<> WaitSignal(RankId peer, int op_cnt, const SignalDesc& signal_desc,
                      const Executor& executor) final;

  absl::Status LaunchAllReduce(se::DeviceAddressBase send_buffer,
                               se::DeviceAddressBase recv_buffer,
                               PrimitiveType dtype, size_t count,
                               ReductionKind reduction_kind,
                               const Executor& executor) final;

  absl::Status LaunchBroadcast(se::DeviceAddressBase send_buffer,
                               se::DeviceAddressBase recv_buffer,
                               PrimitiveType dtype, size_t count, RankId root,
                               const Executor& executor) final;

  absl::Status LaunchReduceScatter(se::DeviceAddressBase send_buffer,
                                   se::DeviceAddressBase recv_buffer,
                                   PrimitiveType dtype, size_t count,
                                   ReductionKind reduction_kind,
                                   const Executor& executor) final;

  absl::Status LaunchAllGather(se::DeviceAddressBase send_buffer,
                               se::DeviceAddressBase recv_buffer,
                               PrimitiveType dtype, size_t count,
                               const Executor& executor) final;

  absl::Status LaunchAllToAll(
      absl::InlinedVector<se::DeviceAddressBase, 4> send_buffers,
      absl::InlinedVector<se::DeviceAddressBase, 4> recv_buffers,
      PrimitiveType dtype, size_t count, const Executor& executor) final;

  absl::Status LaunchCollectivePermute(
      se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
      PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
      absl::Span<const RankId> target_ranks, const Executor& executor) final;

  absl::Status LaunchSend(se::DeviceAddressBase send_buffer,
                          PrimitiveType dtype, size_t count, RankId peer,
                          const Executor& executor) final;

  absl::Status LaunchRecv(se::DeviceAddressBase recv_buffer,
                          PrimitiveType dtype, size_t count, RankId peer,
                          const Executor& executor) final;

  absl::StatusOr<ccl::communicator> Split(int32_t color, int32_t key,
                                          bool split_external_use) const;

  std::string ToString() const final;

  const ccl::communicator& comm() const { return *comm_; }
  std::shared_ptr<ccl::kvs_interface> kvs() const { return kvs_; }

 private:
  OnecclCommunicator(se::StreamExecutor* stream_executor,
                     ccl::communicator comm,
                     std::shared_ptr<ccl::kvs_interface> kvs,
                     std::unique_ptr<tsl::Executor> executor,
                     std::shared_ptr<CancellationToken> cancel);

  absl::Status GroupLaunch(absl::FunctionRef<absl::Status()> group);
  absl::Status CheckReady() const;
  Future<> Execute(absl::AnyInvocable<absl::Status() &&> f) const;

  template <typename T>
  Future<T> Execute(absl::AnyInvocable<absl::StatusOr<T>() &&> f) const;

  absl::Status ExecuteAwait(absl::AnyInvocable<absl::Status() &&> f) const {
    return Execute(std::move(f)).Await();
  }

  template <typename T>
  absl::StatusOr<T> ExecuteAwait(
      absl::AnyInvocable<absl::StatusOr<T>() &&> f) const {
    return Execute<T>(std::move(f)).Await();
  }

  se::StreamExecutor* stream_executor_;
  std::unique_ptr<ccl::communicator> comm_;
  std::shared_ptr<ccl::kvs_interface> kvs_;
  std::unique_ptr<tsl::Executor> executor_;
  std::shared_ptr<CancellationToken> cancel_;
  bool aborted_ = false;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_ONECCL_COMMUNICATOR_H_
