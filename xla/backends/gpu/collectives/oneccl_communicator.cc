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

#include "xla/backends/gpu/collectives/oneccl_communicator.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "oneapi/ccl.hpp"
#include "sycl/sycl.hpp"
#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/collectives/oneccl_registered_memory.h"
#include "xla/backends/gpu/collectives/oneccl_symmetric_memory.h"
#include "xla/backends/gpu/collectives/single_threaded_executor.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/core/collectives/registered_memory.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/future.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

se::Stream* ToStream(const Communicator::Executor& executor) {
  return absl::down_cast<const GpuCollectives::Executor&>(executor).stream();
}

absl::Status OnecclExceptionStatus(absl::string_view expr,
                                   absl::string_view message) {
  return absl::InternalError(
      absl::StrFormat("oneCCL call failed: %s: %s", expr, message));
}

absl::Status OnecclCall(absl::string_view expr,
                        absl::AnyInvocable<void() &&> fn) {
  try {
    std::move(fn)();
    return absl::OkStatus();
  } catch (const ccl::exception& e) {
    return OnecclExceptionStatus(expr, e.what());
  } catch (const std::exception& e) {
    return OnecclExceptionStatus(expr, e.what());
  } catch (...) {
    return OnecclExceptionStatus(expr, "unknown exception");
  }
}

bool OnecclCollectivePermuteBypassSyclP2P() {
  static const bool enabled = [] {
    const char* value =
        std::getenv("XLA_ONECCL_COLLECTIVE_PERMUTE_BYPASS_SYCL_P2P");
    if (value == nullptr) {
      return false;
    }
    absl::string_view v(value);
    return !(v.empty() || v == "0" || v == "false" || v == "False" ||
             v == "FALSE");
  }();
  return enabled;
}

template <typename T, typename F>
absl::StatusOr<T> OnecclValue(absl::string_view expr, F&& fn) {
  try {
    return std::forward<F>(fn)();
  } catch (const ccl::exception& e) {
    return OnecclExceptionStatus(expr, e.what());
  } catch (const std::exception& e) {
    return OnecclExceptionStatus(expr, e.what());
  } catch (...) {
    return OnecclExceptionStatus(expr, "unknown exception");
  }
}

absl::Status VerifyStreamExecutor(se::Stream* stream,
                                  se::StreamExecutor* stream_executor) {
  if (stream == nullptr) {
    return InvalidArgument("oneCCL collective requires a non-null stream");
  }
  if (stream->parent() != stream_executor) {
    return InvalidArgument(
        "oneCCL communicator for device %d cannot launch on a stream for "
        "device %d",
        stream_executor->device_ordinal(), stream->parent()->device_ordinal());
  }
  return absl::OkStatus();
}

absl::StatusOr<ccl::stream> ToOnecclStream(se::Stream* stream) {
  void* handle = stream->platform_specific_handle().stream;
  if (handle == nullptr) {
    return InvalidArgument("oneCCL collective requires a SYCL queue stream");
  }

  auto* queue = static_cast<::sycl::queue*>(handle);
  struct StreamStore {
    absl::Mutex mu;
    absl::flat_hash_map<::sycl::queue*, std::shared_ptr<ccl::stream>> streams
        ABSL_GUARDED_BY(mu);
  };
  static auto* store = new StreamStore;

  absl::MutexLock lock(&store->mu);
  auto it = store->streams.find(queue);
  if (it != store->streams.end()) {
    return *it->second;
  }

  ASSIGN_OR_RETURN(ccl::stream ccl_stream,
                   OnecclValue<ccl::stream>(
                       "ccl::create_stream",
                       [&] { return ccl::create_stream(*queue); }));
  auto shared_stream = std::make_shared<ccl::stream>(std::move(ccl_stream));
  store->streams.emplace(queue, shared_stream);
  return *shared_stream;
}

size_t ToOnecclCount(PrimitiveType dtype, size_t count) {
  return primitive_util::IsComplexType(dtype) ? count * 2 : count;
}

absl::StatusOr<ccl::datatype> ToOnecclDataType(PrimitiveType dtype) {
  switch (dtype) {
    case S8:
      return ccl::datatype::int8;
    case PRED:
    case U8:
      return ccl::datatype::uint8;
    case S16:
      return ccl::datatype::int16;
    case U16:
      return ccl::datatype::uint16;
    case S32:
      return ccl::datatype::int32;
    case U32:
      return ccl::datatype::uint32;
    case S64:
      return ccl::datatype::int64;
    case U64:
      return ccl::datatype::uint64;
    case F16:
      return ccl::datatype::float16;
    case F32:
    case C64:
      return ccl::datatype::float32;
    case F64:
    case C128:
      return ccl::datatype::float64;
    case BF16:
      return ccl::datatype::bfloat16;
    default:
      return InvalidArgument("Unsupported oneCCL data type: %s",
                             primitive_util::LowercasePrimitiveTypeName(dtype));
  }
}

ccl::reduction ToOnecclReduction(ReductionKind kind) {
  switch (kind) {
    case ReductionKind::SUM:
      return ccl::reduction::sum;
    case ReductionKind::PRODUCT:
      return ccl::reduction::prod;
    case ReductionKind::MIN:
      return ccl::reduction::min;
    case ReductionKind::MAX:
      return ccl::reduction::max;
  }
}

std::optional<se::DeviceAddressBase> IsContiguous(
    absl::Span<const se::DeviceAddressBase> buffers) {
  if (buffers.empty()) {
    return std::nullopt;
  }

  if (buffers.size() == 1) {
    return buffers[0];
  }

  size_t total_size = buffers[0].size();
  for (size_t i = 1; i < buffers.size(); ++i) {
    se::DeviceAddress<uint8_t> previous(buffers[i - 1]);
    se::DeviceAddress<uint8_t> current(buffers[i]);
    total_size += current.size();

    if (previous.base() + previous.size() != current.base()) {
      return std::nullopt;
    }
  }

  return se::DeviceAddressBase(buffers[0].opaque(), total_size);
}

}  // namespace

OnecclCommunicator::OnecclCommunicator(
    se::StreamExecutor* stream_executor, ccl::communicator comm,
    std::shared_ptr<ccl::kvs_interface> kvs,
    std::unique_ptr<tsl::Executor> executor,
    std::shared_ptr<CancellationToken> cancel)
    : stream_executor_(stream_executor),
      comm_(std::make_unique<ccl::communicator>(std::move(comm))),
      kvs_(std::move(kvs)),
      executor_(std::move(executor)),
      cancel_(std::move(cancel)) {
  VLOG(1) << absl::StreamFormat("[%d] Created oneCCL communicator %s",
                                stream_executor_->device_ordinal(),
                                ToString());
}

absl::StatusOr<std::unique_ptr<OnecclCommunicator>>
OnecclCommunicator::Create(se::StreamExecutor* stream_executor,
                           ccl::communicator comm,
                           std::shared_ptr<ccl::kvs_interface> kvs,
                           std::shared_ptr<CancellationToken> cancel,
                           tsl::Env& env) {
  if (cancel == nullptr) {
    cancel = std::make_shared<CancellationToken>();
  }
  if (cancel->IsCancelled()) {
    return FailedPrecondition("oneCCL communicator creation cancelled");
  }

  auto executor = std::make_unique<SingleThreadedExecutor>(env);
  return absl::WrapUnique(new OnecclCommunicator(
      stream_executor, std::move(comm), std::move(kvs), std::move(executor),
      std::move(cancel)));
}

OnecclCommunicator::~OnecclCommunicator() {
  absl::Status status = Execute([this]() -> absl::Status {
                          auto activation = stream_executor_->Activate();
                          for (auto& scratch_entry :
                               collective_permute_stream_scratch_) {
                            auto& scratch = scratch_entry.second;
                            if (!scratch.address.is_null()) {
                              stream_executor_->Deallocate(&scratch.address);
                            }
                          }
                          collective_permute_stream_scratch_.clear();
                          if (comm_ == nullptr || aborted_) {
                            return absl::OkStatus();
                          }
                          VLOG(1) << "Destroy oneCCL communicator: "
                                  << ToString();
                          comm_.reset();
                          return absl::OkStatus();
                        }).Await();
  if (!status.ok()) {
    LOG(ERROR) << "Failed to destroy oneCCL communicator: " << status;
  }
}

absl::Status OnecclCommunicator::CheckReady() const {
  if (cancel_->IsCancelled() || aborted_) {
    return FailedPrecondition("OnecclCommunicator aborted");
  }
  if (comm_ == nullptr) {
    return FailedPrecondition("OnecclCommunicator has no underlying comm");
  }
  return absl::OkStatus();
}

absl::Status OnecclCommunicator::Abort() {
  cancel_->Cancel();

  return ExecuteAwait([this]() -> absl::Status {
    if (comm_ == nullptr || aborted_) {
      return FailedPrecondition("OnecclCommunicator already aborted");
    }

    auto activation = stream_executor_->Activate();
    VLOG(1) << "Abort oneCCL communicator: " << ToString();
    comm_.reset();
    aborted_ = true;
    return absl::OkStatus();
  });
}

absl::Status OnecclCommunicator::HealthCheck() const {
  return ExecuteAwait([this]() -> absl::Status {
    auto activation = stream_executor_->Activate();
    return CheckReady();
  });
}

absl::Status OnecclCommunicator::Barrier(const Executor& executor) {
  return ExecuteAwait([this, &executor]() -> absl::Status {
    RETURN_IF_ERROR(CheckReady());
    se::Stream* stream = ToStream(executor);
    RETURN_IF_ERROR(VerifyStreamExecutor(stream, stream_executor_));
    ASSIGN_OR_RETURN(ccl::stream ccl_stream, ToOnecclStream(stream));
    auto activation = stream_executor_->Activate();
    return OnecclCall("ccl::barrier",
                      [&] { ccl::barrier(comm(), ccl_stream); });
  });
}

absl::StatusOr<size_t> OnecclCommunicator::NumRanks() const {
  return ExecuteAwait<size_t>([this]() -> absl::StatusOr<size_t> {
    auto activation = stream_executor_->Activate();
    RETURN_IF_ERROR(CheckReady());
    return comm_->size();
  });
}

absl::StatusOr<size_t> OnecclCommunicator::CurrentRank() {
  return ExecuteAwait<size_t>([this]() -> absl::StatusOr<size_t> {
    auto activation = stream_executor_->Activate();
    RETURN_IF_ERROR(CheckReady());
    return comm_->rank();
  });
}

absl::StatusOr<std::unique_ptr<GpuDeviceCommunicator>>
OnecclCommunicator::CreateDeviceComm(
    const GpuDeviceCommunicator::Requirements& requirements) {
  return Unimplemented(
      "oneCCL does not expose an XLA-compatible device communicator ABI");
}

absl::StatusOr<std::unique_ptr<RegisteredMemory>>
OnecclCommunicator::CreateRegisteredMemory(se::DeviceAddressBase addr) {
  return ExecuteAwait<std::unique_ptr<RegisteredMemory>>(
      [this, addr]() -> absl::StatusOr<std::unique_ptr<RegisteredMemory>> {
        auto activation = stream_executor_->Activate();
        RETURN_IF_ERROR(CheckReady());
        ASSIGN_OR_RETURN(std::unique_ptr<OnecclRegisteredMemory> memory,
                         OnecclRegisteredMemory::Create(comm(), addr));
        return std::unique_ptr<RegisteredMemory>(memory.release());
      });
}

absl::StatusOr<std::unique_ptr<SymmetricMemory>>
OnecclCommunicator::CreateSymmetricMemory(se::DeviceAddressBase addr) {
  return ExecuteAwait<std::unique_ptr<SymmetricMemory>>(
      [this, addr]() -> absl::StatusOr<std::unique_ptr<SymmetricMemory>> {
        auto activation = stream_executor_->Activate();
        RETURN_IF_ERROR(CheckReady());
        ASSIGN_OR_RETURN(std::unique_ptr<OnecclSymmetricMemory> memory,
                         OnecclSymmetricMemory::Create(comm(), addr));
        return std::unique_ptr<SymmetricMemory>(memory.release());
      });
}

absl::Status OnecclCommunicator::GroupLaunch(
    absl::FunctionRef<absl::Status()> group) {
  RETURN_IF_ERROR(CheckReady());
  auto activation = stream_executor_->Activate();
  RETURN_IF_ERROR(OnecclCall("ccl::group_start", [] { ccl::group_start(); }));

  absl::Status status = group();
  absl::Status group_status =
      OnecclCall("ccl::group_end", [] { ccl::group_end(); });
  if (status.ok()) {
    status = group_status;
  }
  return status;
}

absl::StatusOr<se::DeviceAddressBase>
OnecclCommunicator::GetCollectivePermuteScratchForStream(se::Stream* stream,
                                                         size_t bytes) {
  CollectivePermuteScratch& scratch =
      collective_permute_stream_scratch_[stream];
  if (scratch.bytes >= bytes) {
    return scratch.address;
  }
  if (!scratch.address.is_null()) {
    stream_executor_->Deallocate(&scratch.address);
    scratch.bytes = 0;
  }
  scratch.address = stream_executor_->Allocate(bytes, /*memory_space=*/0);
  if (scratch.address.is_null()) {
    return ResourceExhausted(
        "Failed to allocate %d per-stream scratch bytes for oneCCL "
        "collective-permute fallback",
        static_cast<int64_t>(bytes));
  }
  scratch.bytes = bytes;
  return scratch.address;
}

Future<> OnecclCommunicator::GroupExecute(
    absl::AnyInvocable<absl::Status() &&> group) {
  return Execute(std::move(group));
}

Future<> OnecclCommunicator::AllReduce(se::DeviceAddressBase send_buffer,
                                       se::DeviceAddressBase recv_buffer,
                                       PrimitiveType dtype, size_t count,
                                       ReductionKind reduction_kind,
                                       const Executor& executor) {
  return Execute([send_buffer, recv_buffer, dtype, count, reduction_kind,
                  &executor, this]() -> absl::Status {
    return LaunchAllReduce(send_buffer, recv_buffer, dtype, count,
                           reduction_kind, executor);
  });
}

Future<> OnecclCommunicator::Broadcast(se::DeviceAddressBase send_buffer,
                                       se::DeviceAddressBase recv_buffer,
                                       PrimitiveType dtype, size_t count,
                                       RankId root,
                                       const Executor& executor) {
  return Execute([send_buffer, recv_buffer, dtype, count, root, &executor,
                  this]() -> absl::Status {
    return LaunchBroadcast(send_buffer, recv_buffer, dtype, count, root,
                           executor);
  });
}

Future<> OnecclCommunicator::ReduceScatter(se::DeviceAddressBase send_buffer,
                                           se::DeviceAddressBase recv_buffer,
                                           PrimitiveType dtype, size_t count,
                                           ReductionKind reduction_kind,
                                           const Executor& executor) {
  return Execute([send_buffer, recv_buffer, dtype, count, reduction_kind,
                  &executor, this]() -> absl::Status {
    return LaunchReduceScatter(send_buffer, recv_buffer, dtype, count,
                               reduction_kind, executor);
  });
}

Future<> OnecclCommunicator::AllGather(se::DeviceAddressBase send_buffer,
                                       se::DeviceAddressBase recv_buffer,
                                       PrimitiveType dtype, size_t count,
                                       const Executor& executor) {
  return Execute([send_buffer, recv_buffer, dtype, count, &executor,
                  this]() -> absl::Status {
    return LaunchAllGather(send_buffer, recv_buffer, dtype, count, executor);
  });
}

Future<> OnecclCommunicator::AllToAll(
    absl::InlinedVector<se::DeviceAddressBase, 4> send_buffers,
    absl::InlinedVector<se::DeviceAddressBase, 4> recv_buffers,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  return Execute([send_buffers = std::move(send_buffers),
                  recv_buffers = std::move(recv_buffers), dtype, count,
                  &executor, this]() mutable -> absl::Status {
    return LaunchAllToAll(std::move(send_buffers), std::move(recv_buffers),
                          dtype, count, executor);
  });
}

Future<> OnecclCommunicator::CollectivePermute(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
    absl::Span<const RankId> target_ranks, const Executor& executor) {
  std::vector<RankId> owned_target_ranks(target_ranks.begin(),
                                         target_ranks.end());
  return Execute([send_buffer, recv_buffer, dtype, count, source_rank,
                  owned_target_ranks = std::move(owned_target_ranks), &executor,
                  this]() -> absl::Status {
    return LaunchCollectivePermute(send_buffer, recv_buffer, dtype, count,
                                   source_rank, owned_target_ranks, executor);
  });
}

Future<> OnecclCommunicator::Send(se::DeviceAddressBase send_buffer,
                                  PrimitiveType dtype, size_t count,
                                  RankId peer, const Executor& executor) {
  return Execute([send_buffer, dtype, count, peer, &executor,
                  this]() -> absl::Status {
    return LaunchSend(send_buffer, dtype, count, peer, executor);
  });
}

Future<> OnecclCommunicator::Recv(se::DeviceAddressBase recv_buffer,
                                  PrimitiveType dtype, size_t count,
                                  RankId peer, const Executor& executor) {
  return Execute([recv_buffer, dtype, count, peer, &executor,
                  this]() -> absl::Status {
    return LaunchRecv(recv_buffer, dtype, count, peer, executor);
  });
}

Future<> OnecclCommunicator::Put(se::DeviceAddressBase send_buffer,
                                 SymmetricMemory* recv_buffer, size_t offset,
                                 size_t count, RankId peer,
                                 const Executor& executor) {
  return Future<>(Unimplemented(
      "oneCCL Put requires a public XLA-compatible device communicator ABI"));
}

Future<> OnecclCommunicator::Signal(RankId peer,
                                    const SignalDesc& signal_desc,
                                    const Executor& executor) {
  return Future<>(Unimplemented(
      "oneCCL Signal requires a public XLA-compatible device communicator ABI"));
}

Future<> OnecclCommunicator::WaitSignal(RankId peer, int op_cnt,
                                        const SignalDesc& signal_desc,
                                        const Executor& executor) {
  return Future<>(Unimplemented(
      "oneCCL WaitSignal requires a public XLA-compatible device communicator "
      "ABI"));
}

absl::Status OnecclCommunicator::LaunchAllReduce(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Executor& executor) {
  RETURN_IF_ERROR(CheckReady());
  se::Stream* stream = ToStream(executor);
  RETURN_IF_ERROR(VerifyStreamExecutor(stream, stream_executor_));
  ASSIGN_OR_RETURN(ccl::stream ccl_stream, ToOnecclStream(stream));
  auto activation = stream_executor_->Activate();
  ASSIGN_OR_RETURN(ccl::datatype ccl_dtype, ToOnecclDataType(dtype));
  return OnecclCall("ccl::allreduce", [&] {
    ccl::allreduce(send_buffer.opaque(), recv_buffer.opaque(),
                   ToOnecclCount(dtype, count), ccl_dtype,
                   ToOnecclReduction(reduction_kind), comm(), ccl_stream);
  });
}

absl::Status OnecclCommunicator::LaunchBroadcast(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, RankId root, const Executor& executor) {
  RETURN_IF_ERROR(CheckReady());
  se::Stream* stream = ToStream(executor);
  RETURN_IF_ERROR(VerifyStreamExecutor(stream, stream_executor_));
  ASSIGN_OR_RETURN(ccl::stream ccl_stream, ToOnecclStream(stream));
  auto activation = stream_executor_->Activate();
  ASSIGN_OR_RETURN(ccl::datatype ccl_dtype, ToOnecclDataType(dtype));
  return OnecclCall("ccl::broadcast", [&] {
    ccl::broadcast(send_buffer.opaque(), recv_buffer.opaque(),
                   ToOnecclCount(dtype, count), ccl_dtype, root.value(),
                   comm(), ccl_stream);
  });
}

absl::Status OnecclCommunicator::LaunchReduceScatter(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Executor& executor) {
  RETURN_IF_ERROR(CheckReady());
  se::Stream* stream = ToStream(executor);
  RETURN_IF_ERROR(VerifyStreamExecutor(stream, stream_executor_));
  ASSIGN_OR_RETURN(ccl::stream ccl_stream, ToOnecclStream(stream));
  auto activation = stream_executor_->Activate();
  ASSIGN_OR_RETURN(ccl::datatype ccl_dtype, ToOnecclDataType(dtype));
  return OnecclCall("ccl::reduce_scatter", [&] {
    ccl::reduce_scatter(send_buffer.opaque(), recv_buffer.opaque(),
                        ToOnecclCount(dtype, count), ccl_dtype,
                        ToOnecclReduction(reduction_kind), comm(), ccl_stream);
  });
}

absl::Status OnecclCommunicator::LaunchAllGather(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  RETURN_IF_ERROR(CheckReady());
  se::Stream* stream = ToStream(executor);
  RETURN_IF_ERROR(VerifyStreamExecutor(stream, stream_executor_));
  ASSIGN_OR_RETURN(ccl::stream ccl_stream, ToOnecclStream(stream));
  auto activation = stream_executor_->Activate();
  ASSIGN_OR_RETURN(ccl::datatype ccl_dtype, ToOnecclDataType(dtype));
  return OnecclCall("ccl::allgather", [&] {
    ccl::allgather(send_buffer.opaque(), recv_buffer.opaque(),
                   ToOnecclCount(dtype, count), ccl_dtype, comm(), ccl_stream);
  });
}

absl::Status OnecclCommunicator::LaunchAllToAll(
    absl::InlinedVector<se::DeviceAddressBase, 4> send_buffers,
    absl::InlinedVector<se::DeviceAddressBase, 4> recv_buffers,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  RETURN_IF_ERROR(CheckReady());
  if (send_buffers.size() != recv_buffers.size()) {
    return InvalidArgument(
        "Number of send buffers must match number of recv buffers: %d != %d",
        send_buffers.size(), recv_buffers.size());
  }

  se::Stream* stream = ToStream(executor);
  RETURN_IF_ERROR(VerifyStreamExecutor(stream, stream_executor_));
  ASSIGN_OR_RETURN(ccl::stream ccl_stream, ToOnecclStream(stream));
  auto activation = stream_executor_->Activate();

  size_t num_ranks = comm_->size();
  if (send_buffers.size() != num_ranks) {
    return InvalidArgument(
        "Number of send buffers must match number of ranks: %d != %d",
        send_buffers.size(), num_ranks);
  }

  ASSIGN_OR_RETURN(ccl::datatype ccl_dtype, ToOnecclDataType(dtype));

  std::optional<se::DeviceAddressBase> send_contiguous =
      IsContiguous(send_buffers);
  std::optional<se::DeviceAddressBase> recv_contiguous =
      IsContiguous(recv_buffers);
  if (send_contiguous.has_value() && recv_contiguous.has_value()) {
    return OnecclCall("ccl::alltoall", [&] {
      ccl::alltoall(send_contiguous->opaque(), recv_contiguous->opaque(),
                    ToOnecclCount(dtype, count), ccl_dtype, comm(),
                    ccl_stream);
    });
  }

  ccl::vector_class<void*> send_ptrs;
  ccl::vector_class<void*> recv_ptrs;
  send_ptrs.reserve(send_buffers.size());
  recv_ptrs.reserve(recv_buffers.size());
  for (size_t i = 0; i < send_buffers.size(); ++i) {
    send_ptrs.push_back(send_buffers[i].opaque());
    recv_ptrs.push_back(recv_buffers[i].opaque());
  }

  return OnecclCall("ccl::alltoall", [&] {
    ccl::alltoall(send_ptrs, recv_ptrs, ToOnecclCount(dtype, count), ccl_dtype,
                  comm(), ccl_stream);
  });
}

absl::Status OnecclCommunicator::LaunchCollectivePermute(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
    absl::Span<const RankId> target_ranks, const Executor& executor) {
  RETURN_IF_ERROR(CheckReady());
  bool bypass_sycl_p2p = OnecclCollectivePermuteBypassSyclP2P();
  if (!bypass_sycl_p2p && !source_rank.has_value() && target_ranks.empty()) {
    return absl::OkStatus();
  }

  se::Stream* stream = ToStream(executor);
  RETURN_IF_ERROR(VerifyStreamExecutor(stream, stream_executor_));
  ASSIGN_OR_RETURN(ccl::stream ccl_stream, ToOnecclStream(stream));
  auto activation = stream_executor_->Activate();
  ASSIGN_OR_RETURN(ccl::datatype ccl_dtype, ToOnecclDataType(dtype));

  if (bypass_sycl_p2p) {
    if (count == 0) {
      return absl::OkStatus();
    }

    auto* sycl_queue =
        static_cast<::sycl::queue*>(stream->platform_specific_handle().stream);
    if (!sycl_queue->is_in_order()) {
      return FailedPrecondition(
          "oneCCL collective-permute SYCL P2P bypass requires an in-order "
          "SYCL queue");
    }

    int num_ranks = comm().size();
    if (num_ranks == 1 && !source_rank.has_value() && target_ranks.empty()) {
      return absl::OkStatus();
    }
    if (source_rank.has_value() &&
        (source_rank->value() < 0 || source_rank->value() >= num_ranks)) {
      return InvalidArgument("Invalid collective permute source rank: %d",
                             source_rank->value());
    }

    size_t per_rank_bytes = count * primitive_util::ByteWidth(dtype);
    bool single_pair_component =
        num_ranks == 2 && target_ranks.size() <= 1 &&
        (source_rank.has_value() != !target_ranks.empty());
    if (single_pair_component) {
      int64_t root =
          source_rank.has_value() ? source_rank->value() : comm().rank();
      if (root < 0 || root >= num_ranks) {
        return InvalidArgument(
            "Invalid collective permute source rank for two-rank broadcast "
            "fallback: %d",
            root);
      }
      ASSIGN_OR_RETURN(
          se::DeviceAddressBase scratch,
          GetCollectivePermuteScratchForStream(stream, per_rank_bytes));
      void* receive_buffer =
          source_rank.has_value() ? recv_buffer.opaque() : scratch.opaque();
      return OnecclCall("ccl::broadcast collective-permute fallback", [&] {
        ccl::broadcast(send_buffer.opaque(), receive_buffer,
                       ToOnecclCount(dtype, count), ccl_dtype,
                       static_cast<int>(root), comm(), ccl_stream);
      });
    }

    size_t scratch_bytes = static_cast<size_t>(num_ranks) * per_rank_bytes;
    ASSIGN_OR_RETURN(
        se::DeviceAddressBase scratch,
        GetCollectivePermuteScratchForStream(stream, scratch_bytes));
    RETURN_IF_ERROR(
        OnecclCall("ccl::allgather collective-permute fallback", [&] {
          ccl::allgather(send_buffer.opaque(), scratch.opaque(),
                         ToOnecclCount(dtype, count), ccl_dtype, comm(),
                         ccl_stream);
        }));

    if (source_rank.has_value()) {
      se::DeviceAddressBase source_slice = scratch.GetByteSlice(
          static_cast<size_t>(source_rank->value()) * per_rank_bytes,
          per_rank_bytes);
      RETURN_IF_ERROR(
          stream->MemcpyD2D(&recv_buffer, source_slice, per_rank_bytes));
    }
    return absl::OkStatus();
  }

  return GroupLaunch([&]() -> absl::Status {
    if (source_rank.has_value()) {
      RETURN_IF_ERROR(OnecclCall("ccl::recv", [&] {
        ccl::recv(recv_buffer.opaque(), ToOnecclCount(dtype, count), ccl_dtype,
                  source_rank->value(), comm(), ccl_stream);
      }));
    }

    for (RankId target_rank : target_ranks) {
      RETURN_IF_ERROR(OnecclCall("ccl::send", [&] {
        ccl::send(send_buffer.opaque(), ToOnecclCount(dtype, count), ccl_dtype,
                  target_rank.value(), comm(), ccl_stream);
      }));
    }
    return absl::OkStatus();
  });
}

absl::Status OnecclCommunicator::LaunchSend(
    se::DeviceAddressBase send_buffer, PrimitiveType dtype, size_t count,
    RankId peer, const Executor& executor) {
  RETURN_IF_ERROR(CheckReady());
  se::Stream* stream = ToStream(executor);
  RETURN_IF_ERROR(VerifyStreamExecutor(stream, stream_executor_));
  ASSIGN_OR_RETURN(ccl::stream ccl_stream, ToOnecclStream(stream));
  auto activation = stream_executor_->Activate();
  ASSIGN_OR_RETURN(ccl::datatype ccl_dtype, ToOnecclDataType(dtype));
  return OnecclCall("ccl::send", [&] {
    ccl::send(send_buffer.opaque(), ToOnecclCount(dtype, count), ccl_dtype,
              peer.value(), comm(), ccl_stream);
  });
}

absl::Status OnecclCommunicator::LaunchRecv(
    se::DeviceAddressBase recv_buffer, PrimitiveType dtype, size_t count,
    RankId peer, const Executor& executor) {
  RETURN_IF_ERROR(CheckReady());
  se::Stream* stream = ToStream(executor);
  RETURN_IF_ERROR(VerifyStreamExecutor(stream, stream_executor_));
  ASSIGN_OR_RETURN(ccl::stream ccl_stream, ToOnecclStream(stream));
  auto activation = stream_executor_->Activate();
  ASSIGN_OR_RETURN(ccl::datatype ccl_dtype, ToOnecclDataType(dtype));
  return OnecclCall("ccl::recv", [&] {
    ccl::recv(recv_buffer.opaque(), ToOnecclCount(dtype, count), ccl_dtype,
              peer.value(), comm(), ccl_stream);
  });
}

absl::StatusOr<ccl::communicator> OnecclCommunicator::Split(
    int32_t color, int32_t key, bool split_external_use) const {
  return ExecuteAwait<ccl::communicator>(
      [this, color, key, split_external_use]
      () -> absl::StatusOr<ccl::communicator> {
        auto activation = stream_executor_->Activate();
        RETURN_IF_ERROR(CheckReady());
        return OnecclValue<ccl::communicator>("ccl::communicator::split", [&] {
          return comm_->split(color, key, split_external_use);
        });
      });
}

std::string OnecclCommunicator::ToString() const {
  return absl::StrFormat("OnecclCommunicator(comm=%p)", comm_.get());
}

Future<> OnecclCommunicator::Execute(
    absl::AnyInvocable<absl::Status() &&> f) const {
  return Future<>(std::move(f)());
}

template <typename T>
Future<T> OnecclCommunicator::Execute(
    absl::AnyInvocable<absl::StatusOr<T>() &&> f) const {
  return Future<T>(std::move(f)());
}

}  // namespace xla::gpu
