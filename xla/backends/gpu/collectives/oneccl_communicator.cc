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
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

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
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/core/collectives/registered_memory.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/future.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/sycl/sycl_executor.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

absl::StatusOr<se::Stream*> ToStream(
    const Communicator::Executor& executor) {
  auto* gpu_executor = dynamic_cast<const GpuCollectives::Executor*>(&executor);
  if (gpu_executor == nullptr) {
    return InvalidArgument("oneCCL collective requires a GPU executor");
  }
  return gpu_executor->stream();
}

absl::Status OnecclExceptionStatus(absl::string_view expr,
                                   absl::string_view message) {
  return absl::InternalError(
      absl::StrFormat("oneCCL call failed: %s: %s", expr, message));
}

absl::Status WaitForOnecclEvent(absl::string_view expr, ccl::event event) {
  try {
    // XLA marks the collective complete only after oneCCL's returned event is
    // complete, so downstream stream synchronization sees the collective work.
    event.wait();
    return absl::OkStatus();
  } catch (const ccl::exception& e) {
    return OnecclExceptionStatus(expr, e.what());
  } catch (const std::exception& e) {
    return OnecclExceptionStatus(expr, e.what());
  } catch (...) {
    return OnecclExceptionStatus(expr, "unknown exception");
  }
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

absl::Status LaunchOnecclAndWait(absl::string_view expr,
                                 absl::AnyInvocable<ccl::event() &&> launch) {
  ASSIGN_OR_RETURN(ccl::event event,
                   OnecclValue<ccl::event>(expr, std::move(launch)));
  return WaitForOnecclEvent(expr, std::move(event));
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

absl::StatusOr<::sycl::queue*> ToSyclQueue(se::Stream* stream) {
  void* handle = stream->platform_specific_handle().stream;
  if (handle == nullptr) {
    return InvalidArgument("oneCCL collective requires a SYCL queue stream");
  }
  return static_cast<::sycl::queue*>(handle);
}

absl::Status VerifySyclQueueIdentity(
    ::sycl::queue* queue, se::StreamExecutor* stream_executor) {
  auto* sycl_executor =
      dynamic_cast<::stream_executor::sycl::SyclExecutor*>(stream_executor);
  if (sycl_executor == nullptr) {
    return InvalidArgument("oneCCL collective requires a SYCL executor");
  }

  try {
    if (!queue->has_property<::sycl::property::queue::in_order>()) {
      return InvalidArgument(
          "oneCCL collective requires an in-order SYCL queue");
    }

    ::sycl::device queue_device = queue->get_device();
    ::sycl::device executor_device = sycl_executor->GetDevice();
    if (queue_device != executor_device) {
      return InvalidArgument(
          "oneCCL communicator for device %d cannot launch on a SYCL queue "
          "for a different device",
          stream_executor->device_ordinal());
    }

    ::sycl::context queue_context = queue->get_context();
    ASSIGN_OR_RETURN(::sycl::context executor_context,
                     sycl_executor->GetContext());
    if (queue_context != executor_context) {
      return InvalidArgument(
          "oneCCL communicator for device %d cannot launch on a SYCL queue "
          "with a different context",
          stream_executor->device_ordinal());
    }
  } catch (const ::sycl::exception& e) {
    return absl::InternalError(absl::StrFormat(
        "SYCL queue identity check failed for oneCCL launch: %s", e.what()));
  } catch (const std::exception& e) {
    return absl::InternalError(absl::StrFormat(
        "SYCL queue identity check failed for oneCCL launch: %s", e.what()));
  } catch (...) {
    return absl::InternalError(
        "SYCL queue identity check failed for oneCCL launch: unknown "
        "exception");
  }

  return absl::OkStatus();
}

class OnecclStreamCache : public se::Stream::Resource {
 public:
  absl::StatusOr<ccl::stream> Get(::sycl::queue* queue) {
    absl::MutexLock lock(&mu_);
    if (ccl_stream_.has_value() && queue_ == queue) {
      return *ccl_stream_;
    }

    ASSIGN_OR_RETURN(ccl::stream ccl_stream,
                     OnecclValue<ccl::stream>(
                         "ccl::create_stream",
                         [&] { return ccl::create_stream(*queue); }));
    queue_ = queue;
    ccl_stream_.emplace(std::move(ccl_stream));
    return *ccl_stream_;
  }

 private:
  absl::Mutex mu_;
  ::sycl::queue* queue_ ABSL_GUARDED_BY(mu_) = nullptr;
  std::optional<ccl::stream> ccl_stream_ ABSL_GUARDED_BY(mu_);
};

absl::StatusOr<ccl::stream> ToOnecclStream(se::Stream* stream,
                                           ::sycl::queue* queue) {
  OnecclStreamCache* cache =
      stream->GetOrConstructResource<OnecclStreamCache>();
  return cache->Get(queue);
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
    std::shared_ptr<CancellationToken> cancel)
    : stream_executor_(stream_executor),
      comm_(std::make_unique<ccl::communicator>(std::move(comm))),
      kvs_(std::move(kvs)),
      cancel_(std::move(cancel)) {}

absl::StatusOr<std::unique_ptr<OnecclCommunicator>>
OnecclCommunicator::Create(se::StreamExecutor* stream_executor,
                           ccl::communicator comm,
                           std::shared_ptr<ccl::kvs_interface> kvs,
                           std::shared_ptr<CancellationToken> cancel) {
  if (cancel == nullptr) {
    cancel = std::make_shared<CancellationToken>();
  }
  if (cancel->IsCancelled()) {
    return FailedPrecondition("oneCCL communicator creation cancelled");
  }

  return absl::WrapUnique(new OnecclCommunicator(
      stream_executor, std::move(comm), std::move(kvs), std::move(cancel)));
}

OnecclCommunicator::~OnecclCommunicator() {
  absl::Status status = Execute([this]() -> absl::Status {
                          if (comm_ == nullptr || aborted_) {
                            return absl::OkStatus();
                          }
                          auto activation = stream_executor_->Activate();
                          VLOG(2) << "Destroy oneCCL communicator: "
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
    return LaunchOnStream(executor, [&](const ccl::stream& ccl_stream) {
      return LaunchOnecclAndWait("ccl::barrier", [&] {
        return ccl::barrier(comm(), ccl_stream);
      });
    });
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

Future<> OnecclCommunicator::GroupExecute(
    absl::AnyInvocable<absl::Status() &&> group) {
  // oneCCL group_end() only guarantees that grouped operations have been
  // enqueued, and ccl::event::wait() is not supported for collectives issued
  // inside the group API. Execute the body directly so each launched collective
  // can wait its returned event before XLA observes completion.
  return Execute(std::move(group));
}

Future<> OnecclCommunicator::GroupExecuteCounted(
    absl::AnyInvocable<absl::Status() &&> group, int64_t) {
  return GroupExecute(std::move(group));
}

Future<> OnecclCommunicator::AllReduce(se::DeviceAddressBase send_buffer,
                                       se::DeviceAddressBase recv_buffer,
                                       PrimitiveType dtype, size_t count,
                                       ReductionKind reduction_kind,
                                       const Executor& executor) {
  absl::StatusOr<se::Stream*> stream = ToStream(executor);
  if (!stream.ok()) return Future<>(stream.status());
  return Execute([send_buffer, recv_buffer, dtype, count, reduction_kind,
                  stream = *stream, this]() -> absl::Status {
    GpuCollectives::Executor executor(stream);
    return LaunchAllReduce(send_buffer, recv_buffer, dtype, count,
                           reduction_kind, executor);
  });
}

Future<> OnecclCommunicator::Broadcast(se::DeviceAddressBase send_buffer,
                                       se::DeviceAddressBase recv_buffer,
                                       PrimitiveType dtype, size_t count,
                                       RankId root,
                                       const Executor& executor) {
  absl::StatusOr<se::Stream*> stream = ToStream(executor);
  if (!stream.ok()) return Future<>(stream.status());
  return Execute([send_buffer, recv_buffer, dtype, count, root, stream = *stream,
                  this]() -> absl::Status {
    GpuCollectives::Executor executor(stream);
    return LaunchBroadcast(send_buffer, recv_buffer, dtype, count, root,
                           executor);
  });
}

Future<> OnecclCommunicator::ReduceScatter(se::DeviceAddressBase send_buffer,
                                           se::DeviceAddressBase recv_buffer,
                                           PrimitiveType dtype, size_t count,
                                           ReductionKind reduction_kind,
                                           const Executor& executor) {
  absl::StatusOr<se::Stream*> stream = ToStream(executor);
  if (!stream.ok()) return Future<>(stream.status());
  return Execute([send_buffer, recv_buffer, dtype, count, reduction_kind,
                  stream = *stream, this]() -> absl::Status {
    GpuCollectives::Executor executor(stream);
    return LaunchReduceScatter(send_buffer, recv_buffer, dtype, count,
                               reduction_kind, executor);
  });
}

Future<> OnecclCommunicator::AllGather(se::DeviceAddressBase send_buffer,
                                       se::DeviceAddressBase recv_buffer,
                                       PrimitiveType dtype, size_t count,
                                       const Executor& executor) {
  absl::StatusOr<se::Stream*> stream = ToStream(executor);
  if (!stream.ok()) return Future<>(stream.status());
  return Execute([send_buffer, recv_buffer, dtype, count, stream = *stream,
                  this]() -> absl::Status {
    GpuCollectives::Executor executor(stream);
    return LaunchAllGather(send_buffer, recv_buffer, dtype, count, executor);
  });
}

Future<> OnecclCommunicator::AllToAll(
    absl::InlinedVector<se::DeviceAddressBase, 4> send_buffers,
    absl::InlinedVector<se::DeviceAddressBase, 4> recv_buffers,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  absl::StatusOr<se::Stream*> stream = ToStream(executor);
  if (!stream.ok()) return Future<>(stream.status());
  return Execute([send_buffers = std::move(send_buffers),
                  recv_buffers = std::move(recv_buffers), dtype, count,
                  stream = *stream, this]() mutable -> absl::Status {
    GpuCollectives::Executor executor(stream);
    return LaunchAllToAll(std::move(send_buffers), std::move(recv_buffers),
                          dtype, count, executor);
  });
}

Future<> OnecclCommunicator::CollectivePermute(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
    absl::Span<const RankId> target_ranks, const Executor& executor) {
  absl::StatusOr<se::Stream*> stream = ToStream(executor);
  if (!stream.ok()) return Future<>(stream.status());
  std::vector<RankId> owned_target_ranks(target_ranks.begin(),
                                         target_ranks.end());
  return Execute([send_buffer, recv_buffer, dtype, count, source_rank,
                  owned_target_ranks = std::move(owned_target_ranks),
                  stream = *stream, this]() -> absl::Status {
    GpuCollectives::Executor executor(stream);
    return LaunchCollectivePermute(send_buffer, recv_buffer, dtype, count,
                                   source_rank, owned_target_ranks, executor);
  });
}

Future<> OnecclCommunicator::Send(se::DeviceAddressBase send_buffer,
                                  PrimitiveType dtype, size_t count,
                                  RankId peer, const Executor& executor) {
  absl::StatusOr<se::Stream*> stream = ToStream(executor);
  if (!stream.ok()) return Future<>(stream.status());
  return Execute([send_buffer, dtype, count, peer, stream = *stream,
                  this]() -> absl::Status {
    GpuCollectives::Executor executor(stream);
    return LaunchSend(send_buffer, dtype, count, peer, executor);
  });
}

Future<> OnecclCommunicator::Recv(se::DeviceAddressBase recv_buffer,
                                  PrimitiveType dtype, size_t count,
                                  RankId peer, const Executor& executor) {
  absl::StatusOr<se::Stream*> stream = ToStream(executor);
  if (!stream.ok()) return Future<>(stream.status());
  return Execute([recv_buffer, dtype, count, peer, stream = *stream,
                  this]() -> absl::Status {
    GpuCollectives::Executor executor(stream);
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
  return LaunchOnStream(executor, [&](const ccl::stream& ccl_stream) {
    ASSIGN_OR_RETURN(ccl::datatype ccl_dtype, ToOnecclDataType(dtype));
    return LaunchOnecclAndWait("ccl::allreduce", [&] {
      return ccl::allreduce(send_buffer.opaque(), recv_buffer.opaque(),
                            ToOnecclCount(dtype, count), ccl_dtype,
                            ToOnecclReduction(reduction_kind), comm(),
                            ccl_stream);
    });
  });
}

absl::Status OnecclCommunicator::LaunchBroadcast(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, RankId root, const Executor& executor) {
  return LaunchOnStream(executor, [&](const ccl::stream& ccl_stream) {
    ASSIGN_OR_RETURN(ccl::datatype ccl_dtype, ToOnecclDataType(dtype));
    return LaunchOnecclAndWait("ccl::broadcast", [&] {
      return ccl::broadcast(send_buffer.opaque(), recv_buffer.opaque(),
                            ToOnecclCount(dtype, count), ccl_dtype,
                            root.value(), comm(), ccl_stream);
    });
  });
}

absl::Status OnecclCommunicator::LaunchReduceScatter(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Executor& executor) {
  return LaunchOnStream(executor, [&](const ccl::stream& ccl_stream) {
    ASSIGN_OR_RETURN(ccl::datatype ccl_dtype, ToOnecclDataType(dtype));
    return LaunchOnecclAndWait("ccl::reduce_scatter", [&] {
      return ccl::reduce_scatter(send_buffer.opaque(), recv_buffer.opaque(),
                                 ToOnecclCount(dtype, count), ccl_dtype,
                                 ToOnecclReduction(reduction_kind), comm(),
                                 ccl_stream);
    });
  });
}

absl::Status OnecclCommunicator::LaunchAllGather(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  return LaunchOnStream(executor, [&](const ccl::stream& ccl_stream) {
    ASSIGN_OR_RETURN(ccl::datatype ccl_dtype, ToOnecclDataType(dtype));
    return LaunchOnecclAndWait("ccl::allgather", [&] {
      return ccl::allgather(send_buffer.opaque(), recv_buffer.opaque(),
                            ToOnecclCount(dtype, count), ccl_dtype, comm(),
                            ccl_stream);
    });
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

  return LaunchOnStream(executor, [&](const ccl::stream& ccl_stream) {
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
      return LaunchOnecclAndWait("ccl::alltoall", [&] {
        return ccl::alltoall(send_contiguous->opaque(),
                             recv_contiguous->opaque(),
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

    return LaunchOnecclAndWait("ccl::alltoall", [&] {
      return ccl::alltoall(send_ptrs, recv_ptrs,
                           ToOnecclCount(dtype, count), ccl_dtype, comm(),
                           ccl_stream);
    });
  });
}

absl::Status OnecclCommunicator::LaunchCollectivePermute(
    se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
    PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
    absl::Span<const RankId> target_ranks, const Executor& executor) {
  RETURN_IF_ERROR(CheckReady());
  if (!source_rank.has_value() && target_ranks.empty()) {
    return absl::OkStatus();
  }

  return LaunchOnStream(executor, [&](const ccl::stream& ccl_stream) {
    ASSIGN_OR_RETURN(ccl::datatype ccl_dtype, ToOnecclDataType(dtype));

    std::vector<ccl::event> events;
    events.reserve((source_rank.has_value() ? 1 : 0) + target_ranks.size());

    if (source_rank.has_value()) {
      ASSIGN_OR_RETURN(
          ccl::event event, OnecclValue<ccl::event>("ccl::recv", [&] {
            return ccl::recv(recv_buffer.opaque(), ToOnecclCount(dtype, count),
                             ccl_dtype, source_rank->value(), comm(),
                             ccl_stream);
          }));
      events.push_back(std::move(event));
    }

    for (RankId target_rank : target_ranks) {
      ASSIGN_OR_RETURN(ccl::event event,
                       OnecclValue<ccl::event>("ccl::send", [&] {
                         return ccl::send(
                             send_buffer.opaque(), ToOnecclCount(dtype, count),
                             ccl_dtype, target_rank.value(), comm(),
                             ccl_stream);
                       }));
      events.push_back(std::move(event));
    }

    for (ccl::event& event : events) {
      RETURN_IF_ERROR(
          WaitForOnecclEvent("ccl::collective_permute event",
                             std::move(event)));
    }
    return absl::OkStatus();
  });
}

absl::Status OnecclCommunicator::LaunchSend(
    se::DeviceAddressBase send_buffer, PrimitiveType dtype, size_t count,
    RankId peer, const Executor& executor) {
  return LaunchOnStream(executor, [&](const ccl::stream& ccl_stream) {
    ASSIGN_OR_RETURN(ccl::datatype ccl_dtype, ToOnecclDataType(dtype));
    return LaunchOnecclAndWait("ccl::send", [&] {
      return ccl::send(send_buffer.opaque(), ToOnecclCount(dtype, count),
                       ccl_dtype, peer.value(), comm(), ccl_stream);
    });
  });
}

absl::Status OnecclCommunicator::LaunchRecv(
    se::DeviceAddressBase recv_buffer, PrimitiveType dtype, size_t count,
    RankId peer, const Executor& executor) {
  return LaunchOnStream(executor, [&](const ccl::stream& ccl_stream) {
    ASSIGN_OR_RETURN(ccl::datatype ccl_dtype, ToOnecclDataType(dtype));
    return LaunchOnecclAndWait("ccl::recv", [&] {
      return ccl::recv(recv_buffer.opaque(), ToOnecclCount(dtype, count),
                       ccl_dtype, peer.value(), comm(), ccl_stream);
    });
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

absl::Status OnecclCommunicator::LaunchOnStream(
    const Executor& executor,
    absl::AnyInvocable<absl::Status(const ccl::stream&) &&> launch) const {
  RETURN_IF_ERROR(CheckReady());
  ASSIGN_OR_RETURN(se::Stream* stream, ToStream(executor));
  RETURN_IF_ERROR(VerifyStreamExecutor(stream, stream_executor_));
  ASSIGN_OR_RETURN(::sycl::queue* queue, ToSyclQueue(stream));
  RETURN_IF_ERROR(VerifySyclQueueIdentity(queue, stream_executor_));
  ASSIGN_OR_RETURN(ccl::stream ccl_stream, ToOnecclStream(stream, queue));
  auto activation = stream_executor_->Activate();
  return std::move(launch)(ccl_stream);
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
