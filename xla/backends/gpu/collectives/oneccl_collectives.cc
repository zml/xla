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

#include "xla/backends/gpu/collectives/oneccl_collectives.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>
#include "absl/base/call_once.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "oneapi/ccl.h"
#include "oneapi/ccl.hpp"
#include "tsl/platform/casts.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/core/collectives/collectives_registry.h"
#include "xla/future.h"
#include "xla/primitive_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/sycl/sycl_executor.h"
#include "xla/stream_executor/sycl/sycl_stream.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

absl::once_flag ccl_init_once;

void SetOnecclEnvDefault(const char* name, const char* value) {
  if (std::getenv(name) != nullptr) return;
  if (::setenv(name, value, /*overwrite=*/0) != 0) {
    LOG(WARNING) << "Failed to set oneCCL default " << name << "=" << value;
  }
}

void InitCclOnce() {
  // On Arc B-series, oneCCL's large SYCL kernels currently reset or crash the
  // device in the one-thread-per-rank configuration. Keep these collectives on
  // oneCCL's PCIe/LL P2P path unless the user explicitly overrides the choice.
  SetOnecclEnvDefault("CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD", "1073741824");
  SetOnecclEnvDefault("CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD", "33554432");
  ccl::init();
}

absl::Status EnsureCclInitialized() {
  try {
    absl::call_once(ccl_init_once, InitCclOnce);
  } catch (const std::exception& e) {
    return Internal("oneCCL initialization failed: %s", e.what());
  }
  return absl::OkStatus();
}

std::string CliqueIdKey(const CliqueId& id) {
  return std::string(id.data().begin(), id.data().end());
}

struct KvsStore {
  absl::Mutex mu;
  absl::flat_hash_map<std::string, ccl::shared_ptr_class<ccl::kvs>> main_kvs
      ABSL_GUARDED_BY(mu);
};

KvsStore& ProcessKvsStore() {
  static auto* store = new KvsStore;
  return *store;
}

struct StreamStore {
  absl::Mutex mu;
  absl::flat_hash_map<::sycl::queue*, std::shared_ptr<ccl::stream>> streams
      ABSL_GUARDED_BY(mu);
};

StreamStore& ProcessStreamStore() {
  static auto* store = new StreamStore;
  return *store;
}

absl::StatusOr<ccl::stream> GetOrCreateCclStream(::sycl::queue* queue) {
  absl::MutexLock lock(&ProcessStreamStore().mu);
  auto it = ProcessStreamStore().streams.find(queue);
  if (it != ProcessStreamStore().streams.end()) {
    return *it->second;
  }

  try {
    auto stream = std::make_shared<ccl::stream>(ccl::create_stream(*queue));
    auto [inserted, _] = ProcessStreamStore().streams.emplace(queue, stream);
    return *inserted->second;
  } catch (const std::exception& e) {
    return Internal("oneCCL stream creation failed: %s", e.what());
  }
}

bool LogOnecclCollectives() {
  const char* value = std::getenv("XLA_SYCL_LOG_ONECCL_COLLECTIVES");
  if (value == nullptr) return false;
  std::string flag(value);
  return flag == "1" || flag == "true" || flag == "TRUE";
}

bool BlockHostOnOnecclCollectives() {
  const char* value = std::getenv("XLA_SYCL_ONECCL_BLOCKING_WAIT");
  if (value == nullptr) return false;
  std::string flag(value);
  return flag == "1" || flag == "true" || flag == "TRUE";
}

void MaybeWaitForOnecclEvent(ccl::event event) {
  if (BlockHostOnOnecclCollectives()) {
    event.wait();
  }
}

void LogOnecclCollective(const char* op, int rank, int comm_size,
                         PrimitiveType dtype, size_t count, size_t bytes) {
  if (!LogOnecclCollectives()) return;
  static std::atomic<uint64_t> logged{0};
  uint64_t log_index = logged.fetch_add(1);
  if (log_index >= 400) return;
  std::string dtype_name = primitive_util::LowercasePrimitiveTypeName(dtype);
  std::fprintf(stderr,
               "oneCCL collective #%llu op=%s rank=%d/%d dtype=%s count=%zu "
               "bytes=%zu\n",
               static_cast<unsigned long long>(log_index), op, rank,
               comm_size, dtype_name.c_str(), count, bytes);
}

absl::StatusOr<ccl::shared_ptr_class<ccl::kvs>> CreateKvs(
    const std::optional<CliqueIds>& clique_ids) {
  TF_RETURN_IF_ERROR(EnsureCclInitialized());

  if (!clique_ids.has_value() || clique_ids->size() == 0) {
    return ccl::create_main_kvs();
  }

  const CliqueId& clique_id = clique_ids->at(0);
  std::string key = CliqueIdKey(clique_id);
  {
    absl::MutexLock lock(&ProcessKvsStore().mu);
    auto it = ProcessKvsStore().main_kvs.find(key);
    if (it != ProcessKvsStore().main_kvs.end()) {
      return it->second;
    }
  }

  if (clique_id.size() != ccl::kvs::address_max_size) {
    return InvalidArgument("oneCCL CliqueId size must be %d bytes, got %d",
                           ccl::kvs::address_max_size, clique_id.size());
  }

  ccl::kvs::address_type address;
  std::copy(clique_id.data().begin(), clique_id.data().end(),
            address.begin());
  return ccl::create_kvs(address);
}

absl::StatusOr<ccl::datatype> ToCclDataType(PrimitiveType dtype,
                                            bool is_reduction_op) {
  switch (dtype) {
    case S8:
    case F8E5M2:
    case F8E4M3FN:
    case F8E5M2FNUZ:
    case F8E4M3FNUZ:
    case F8E8M0FNU:
      if (is_reduction_op && dtype != S8) {
        return InvalidArgument("Unsupported data type for oneCCL reduction: %s",
                               primitive_util::LowercasePrimitiveTypeName(
                                   dtype));
      }
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

size_t ToCclCount(PrimitiveType dtype, size_t count) {
  return primitive_util::IsComplexType(dtype) ? count * 2 : count;
}

absl::StatusOr<ccl::reduction> ToCclReduction(ReductionKind kind) {
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

absl::StatusOr<se::Stream*> ToStream(const Communicator::Executor& executor) {
  auto* gpu_executor = tsl::down_cast<const GpuCollectives::Executor*>(
      &executor);
  if (gpu_executor == nullptr) {
    return InvalidArgument("Expected GpuCollectives::Executor");
  }
  return gpu_executor->stream();
}

absl::StatusOr<::sycl::queue*> ToSyclQueue(se::Stream* stream) {
  auto* sycl_stream =
      tsl::down_cast<stream_executor::sycl::SyclStream*>(stream);
  if (sycl_stream == nullptr) {
    return InvalidArgument("Expected SYCL stream");
  }
  return sycl_stream->stream_handle();
}

absl::StatusOr<ccl::stream> ToCclStream(
    const Communicator::Executor& executor) {
  TF_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));
  TF_ASSIGN_OR_RETURN(::sycl::queue * queue, ToSyclQueue(stream));
  return GetOrCreateCclStream(queue);
}

absl::StatusOr<stream_executor::sycl::SyclExecutor*> ToSyclExecutor(
    const Collectives::DeviceRank& rank) {
  auto* device = tsl::down_cast<GpuCollectives::Device*>(rank.device);
  if (device == nullptr) {
    return InvalidArgument("Expected GpuCollectives::Device");
  }
  auto* sycl_executor =
      tsl::down_cast<stream_executor::sycl::SyclExecutor*>(
          device->stream_executor());
  if (sycl_executor == nullptr) {
    return InvalidArgument("Expected SYCL stream executor");
  }
  return sycl_executor;
}

template <typename F>
absl::Status RunCcl(F&& f) {
  try {
    std::forward<F>(f)();
  } catch (const std::exception& e) {
    return Internal("oneCCL operation failed: %s", e.what());
  }
  return absl::OkStatus();
}

class OnecclCommunicator : public GpuCommunicator {
 public:
  OnecclCommunicator(se::StreamExecutor* stream_executor, ccl::communicator comm)
      : stream_executor_(stream_executor), comm_(std::move(comm)) {}

  absl::Status Abort() final { return absl::OkStatus(); }
  absl::Status HealthCheck() const final { return absl::OkStatus(); }

  absl::StatusOr<size_t> NumRanks() const final { return comm_.size(); }
  absl::StatusOr<size_t> CurrentRank() final { return comm_.rank(); }

  PlatformCommunicatorHandle platform_comm() const final {
    return PlatformCommunicatorHandle{const_cast<ccl::communicator*>(&comm_)};
  }

  Future<> GroupExecute(
      absl::AnyInvocable<absl::Status(GpuCommunicator*)> f) final {
    return Future<>(f(this));
  }

  Future<> AllReduce(se::DeviceAddressBase send_buffer,
                     se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
                     size_t count, ReductionKind reduction_kind,
                     const Executor& executor) final {
    return Future<>(LaunchAllReduce(send_buffer, recv_buffer, dtype, count,
                                    reduction_kind, executor));
  }

  Future<> Broadcast(se::DeviceAddressBase send_buffer,
                     se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
                     size_t count, RankId root, const Executor& executor) final {
    return Future<>(
        LaunchBroadcast(send_buffer, recv_buffer, dtype, count, root, executor));
  }

  Future<> ReduceScatter(se::DeviceAddressBase send_buffer,
                         se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
                         size_t count, ReductionKind reduction_kind,
                         const Executor& executor) final {
    return Future<>(LaunchReduceScatter(send_buffer, recv_buffer, dtype, count,
                                        reduction_kind, executor));
  }

  Future<> AllGather(se::DeviceAddressBase send_buffer,
                     se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
                     size_t count, const Executor& executor) final {
    return Future<>(
        LaunchAllGather(send_buffer, recv_buffer, dtype, count, executor));
  }

  Future<> AllToAll(absl::InlinedVector<se::DeviceAddressBase, 4> send_buffers,
                    absl::InlinedVector<se::DeviceAddressBase, 4> recv_buffers,
                    PrimitiveType dtype, size_t count,
                    const Executor& executor) final {
    return Future<>(LaunchAllToAll(std::move(send_buffers),
                                   std::move(recv_buffers), dtype, count,
                                   executor));
  }

  Future<> CollectivePermute(se::DeviceAddressBase send_buffer,
                             se::DeviceAddressBase recv_buffer,
                             PrimitiveType dtype, size_t count,
                             std::optional<RankId> source_rank,
                             absl::Span<const RankId> target_ranks,
                             const Executor& executor) final {
    return Future<>(LaunchCollectivePermute(send_buffer, recv_buffer, dtype,
                                            count, source_rank, target_ranks,
                                            executor));
  }

  Future<> Send(se::DeviceAddressBase send_buffer, PrimitiveType dtype,
                size_t count, RankId peer, const Executor& executor) final {
    return Future<>(LaunchSend(send_buffer, dtype, count, peer, executor));
  }

  Future<> Recv(se::DeviceAddressBase recv_buffer, PrimitiveType dtype,
                size_t count, RankId peer, const Executor& executor) final {
    return Future<>(LaunchRecv(recv_buffer, dtype, count, peer, executor));
  }

  std::string ToString() const final {
    return absl::StrFormat("oneCCL communicator rank %d of %d on device %d",
                           comm_.rank(), comm_.size(),
                           stream_executor_->device_ordinal());
  }

  const ccl::communicator& comm() const { return comm_; }

 private:
  absl::Status LaunchAllReduce(se::DeviceAddressBase send_buffer,
                               se::DeviceAddressBase recv_buffer,
                               PrimitiveType dtype, size_t count,
                               ReductionKind reduction_kind,
                               const Executor& executor) final {
    TF_ASSIGN_OR_RETURN(ccl::stream stream, ToCclStream(executor));
    TF_ASSIGN_OR_RETURN(ccl::datatype ccl_dtype,
                        ToCclDataType(dtype, /*is_reduction_op=*/true));
    TF_ASSIGN_OR_RETURN(ccl::reduction reduction,
                        ToCclReduction(reduction_kind));
    LogOnecclCollective("all-reduce", comm_.rank(), comm_.size(), dtype, count,
                        primitive_util::ByteWidth(dtype) * count);
    count = ToCclCount(dtype, count);
    return RunCcl([&] {
      MaybeWaitForOnecclEvent(ccl::allreduce(
          send_buffer.opaque(), recv_buffer.opaque(), count, ccl_dtype,
          reduction, comm_, stream));
    });
  }

  absl::Status LaunchBroadcast(se::DeviceAddressBase send_buffer,
                               se::DeviceAddressBase recv_buffer,
                               PrimitiveType dtype, size_t count, RankId root,
                               const Executor& executor) final {
    TF_ASSIGN_OR_RETURN(se::Stream * xla_stream, ToStream(executor));
    TF_ASSIGN_OR_RETURN(ccl::stream stream, ToCclStream(executor));
    TF_ASSIGN_OR_RETURN(ccl::datatype ccl_dtype,
                        ToCclDataType(dtype, /*is_reduction_op=*/false));
    if (comm_.rank() == root.value() &&
        send_buffer.opaque() != recv_buffer.opaque()) {
      se::DeviceAddressBase dst = recv_buffer;
      TF_RETURN_IF_ERROR(xla_stream->Memcpy(
          &dst, send_buffer, primitive_util::ByteWidth(dtype) * count));
    }
    LogOnecclCollective("broadcast", comm_.rank(), comm_.size(), dtype, count,
                        primitive_util::ByteWidth(dtype) * count);
    count = ToCclCount(dtype, count);
    return RunCcl([&] {
      MaybeWaitForOnecclEvent(ccl::broadcast(
          recv_buffer.opaque(), count, ccl_dtype, root.value(), comm_, stream));
    });
  }

  absl::Status LaunchReduceScatter(se::DeviceAddressBase send_buffer,
                                   se::DeviceAddressBase recv_buffer,
                                   PrimitiveType dtype, size_t count,
                                   ReductionKind reduction_kind,
                                   const Executor& executor) final {
    TF_ASSIGN_OR_RETURN(ccl::stream stream, ToCclStream(executor));
    TF_ASSIGN_OR_RETURN(ccl::datatype ccl_dtype,
                        ToCclDataType(dtype, /*is_reduction_op=*/true));
    TF_ASSIGN_OR_RETURN(ccl::reduction reduction,
                        ToCclReduction(reduction_kind));
    LogOnecclCollective("reduce-scatter", comm_.rank(), comm_.size(), dtype,
                        count, primitive_util::ByteWidth(dtype) * count);
    count = ToCclCount(dtype, count);
    return RunCcl([&] {
      MaybeWaitForOnecclEvent(ccl::reduce_scatter(
          send_buffer.opaque(), recv_buffer.opaque(), count, ccl_dtype,
          reduction, comm_, stream));
    });
  }

  absl::Status LaunchAllGather(se::DeviceAddressBase send_buffer,
                               se::DeviceAddressBase recv_buffer,
                               PrimitiveType dtype, size_t count,
                               const Executor& executor) final {
    TF_ASSIGN_OR_RETURN(ccl::stream stream, ToCclStream(executor));
    TF_ASSIGN_OR_RETURN(ccl::datatype ccl_dtype,
                        ToCclDataType(dtype, /*is_reduction_op=*/false));
    LogOnecclCollective("all-gather", comm_.rank(), comm_.size(), dtype, count,
                        primitive_util::ByteWidth(dtype) * count *
                            comm_.size());
    count = ToCclCount(dtype, count);
    return RunCcl([&] {
      MaybeWaitForOnecclEvent(ccl::allgather(
          send_buffer.opaque(), recv_buffer.opaque(), count, ccl_dtype, comm_,
          stream));
    });
  }

  absl::Status LaunchAllToAll(
      absl::InlinedVector<se::DeviceAddressBase, 4> send_buffers,
      absl::InlinedVector<se::DeviceAddressBase, 4> recv_buffers,
      PrimitiveType dtype, size_t count, const Executor& executor) final {
    if (send_buffers.size() != recv_buffers.size()) {
      return InvalidArgument("AllToAll send/recv buffer counts differ: %d vs %d",
                             send_buffers.size(), recv_buffers.size());
    }
    TF_ASSIGN_OR_RETURN(ccl::stream stream, ToCclStream(executor));
    TF_ASSIGN_OR_RETURN(ccl::datatype ccl_dtype,
                        ToCclDataType(dtype, /*is_reduction_op=*/false));
    ccl::vector_class<void*> ccl_send_buffers;
    ccl::vector_class<void*> ccl_recv_buffers;
    ccl_send_buffers.reserve(send_buffers.size());
    ccl_recv_buffers.reserve(recv_buffers.size());
    for (se::DeviceAddressBase buffer : send_buffers) {
      ccl_send_buffers.push_back(buffer.opaque());
    }
    for (se::DeviceAddressBase buffer : recv_buffers) {
      ccl_recv_buffers.push_back(buffer.opaque());
    }
    LogOnecclCollective("all-to-all", comm_.rank(), comm_.size(), dtype, count,
                        primitive_util::ByteWidth(dtype) * count);
    count = ToCclCount(dtype, count);
    return RunCcl([&] {
      MaybeWaitForOnecclEvent(ccl::alltoall(
          ccl_send_buffers, ccl_recv_buffers, count, ccl_dtype, comm_, stream));
    });
  }

  absl::Status LaunchCollectivePermute(
      se::DeviceAddressBase send_buffer, se::DeviceAddressBase recv_buffer,
      PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
      absl::Span<const RankId> target_ranks, const Executor& executor) final {
    TF_ASSIGN_OR_RETURN(se::Stream * xla_stream, ToStream(executor));
    TF_ASSIGN_OR_RETURN(ccl::stream stream, ToCclStream(executor));
    TF_ASSIGN_OR_RETURN(ccl::datatype ccl_dtype,
                        ToCclDataType(dtype, /*is_reduction_op=*/false));
    if (!source_rank.has_value()) {
      se::DeviceAddressBase dst = recv_buffer;
      TF_RETURN_IF_ERROR(xla_stream->MemZero(
          &dst, primitive_util::ByteWidth(dtype) * count));
    }
    LogOnecclCollective("collective-permute", comm_.rank(), comm_.size(), dtype,
                        count, primitive_util::ByteWidth(dtype) * count);
    count = ToCclCount(dtype, count);
    return RunCcl([&] {
      std::vector<ccl::event> events;
      events.reserve(target_ranks.size() + (source_rank.has_value() ? 1 : 0));
      ccl::group_start();
      for (RankId target : target_ranks) {
        events.push_back(ccl::send(send_buffer.opaque(), count, ccl_dtype,
                                   target.value(), comm_, stream));
      }
      if (source_rank.has_value()) {
        events.push_back(ccl::recv(recv_buffer.opaque(), count, ccl_dtype,
                                   source_rank->value(), comm_, stream));
      }
      ccl::group_end();
      for (ccl::event& event : events) {
        MaybeWaitForOnecclEvent(std::move(event));
      }
    });
  }

  absl::Status LaunchSend(se::DeviceAddressBase send_buffer,
                          PrimitiveType dtype, size_t count, RankId peer,
                          const Executor& executor) final {
    TF_ASSIGN_OR_RETURN(ccl::stream stream, ToCclStream(executor));
    TF_ASSIGN_OR_RETURN(ccl::datatype ccl_dtype,
                        ToCclDataType(dtype, /*is_reduction_op=*/false));
    count = ToCclCount(dtype, count);
    return RunCcl([&] {
      MaybeWaitForOnecclEvent(ccl::send(send_buffer.opaque(), count, ccl_dtype,
                                        peer.value(), comm_, stream));
    });
  }

  absl::Status LaunchRecv(se::DeviceAddressBase recv_buffer,
                          PrimitiveType dtype, size_t count, RankId peer,
                          const Executor& executor) final {
    TF_ASSIGN_OR_RETURN(ccl::stream stream, ToCclStream(executor));
    TF_ASSIGN_OR_RETURN(ccl::datatype ccl_dtype,
                        ToCclDataType(dtype, /*is_reduction_op=*/false));
    count = ToCclCount(dtype, count);
    return RunCcl([&] {
      MaybeWaitForOnecclEvent(ccl::recv(recv_buffer.opaque(), count, ccl_dtype,
                                        peer.value(), comm_, stream));
    });
  }

  se::StreamExecutor* stream_executor_;
  ccl::communicator comm_;
};

absl::StatusOr<std::unique_ptr<Communicator>> Cast(
    absl::StatusOr<std::unique_ptr<OnecclCommunicator>> comm_or) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<OnecclCommunicator> comm,
                      std::move(comm_or));
  return std::unique_ptr<Communicator>(comm.release());
}

absl::StatusOr<std::unique_ptr<OnecclCommunicator>> CreateOnecclCommunicator(
    const Collectives::DeviceRank& rank, int num_ranks,
    ccl::shared_ptr_class<ccl::kvs_interface> kvs) {
  TF_ASSIGN_OR_RETURN(auto* executor, ToSyclExecutor(rank));
  ::sycl::device device = executor->GetDevice();
  TF_ASSIGN_OR_RETURN(::sycl::context context, executor->GetContext());
  ccl::device ccl_device = ccl::create_device(device);
  ccl::context ccl_context = ccl::create_context(context);
  ccl::communicator comm = ccl::create_communicatorExt(
      num_ranks, rank.rank.value(), ccl_device, ccl_context, kvs);
  return std::make_unique<OnecclCommunicator>(executor, std::move(comm));
}

const OnecclCommunicator* Cast(const Communicator* comm) {
  auto* oneccl_comm = tsl::down_cast<const OnecclCommunicator*>(comm);
  CHECK(oneccl_comm != nullptr) << "Unsupported XLA communicator";
  return oneccl_comm;
}

}  // namespace

absl::StatusOr<CliqueId> OnecclCollectives::CreateUniqueCliqueId() const {
  TF_RETURN_IF_ERROR(EnsureCclInitialized());

  ccl::shared_ptr_class<ccl::kvs> kvs = ccl::create_main_kvs();
  ccl::kvs::address_type address = kvs->get_address();
  std::string key(address.begin(), address.end());
  {
    absl::MutexLock lock(&ProcessKvsStore().mu);
    ProcessKvsStore().main_kvs.emplace(key, kvs);
  }
  return CliqueId(key);
}

absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
OnecclCollectives::CreateCommunicators(
    const CliqueKey& clique_key, const std::optional<CliqueIds>& clique_ids,
    absl::Span<const DeviceRank> ranks, const Collectives::Config& config) {
  VLOG(1) << absl::StreamFormat(
      "Initialize oneCCL communicators for %d local ranks out of %d total "
      "ranks",
      ranks.size(), clique_key.num_devices());

  if (ranks.empty()) {
    return std::vector<std::unique_ptr<Communicator>>();
  }

  TF_ASSIGN_OR_RETURN(ccl::shared_ptr_class<ccl::kvs> kvs,
                      CreateKvs(clique_ids));

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "CreateOnecclComms",
                               ranks.size());

  std::vector<Future<std::unique_ptr<Communicator>>> futures(ranks.size());
  for (size_t i = 0; i < ranks.size(); ++i) {
    futures[i] = MakeFutureOn(*pool.AsExecutor(), [&, i] {
      absl::StatusOr<std::unique_ptr<OnecclCommunicator>> comm =
          CreateOnecclCommunicator(ranks[i], clique_key.num_devices(), kvs);
      if (!comm.ok()) {
        LOG(ERROR) << absl::StreamFormat(
            "[rank=%v] Failed to create oneCCL communicator: %s",
            ranks[i].rank, comm.status().ToString());
      }
      return Cast(std::move(comm));
    });
  }

  return JoinFutures(absl::MakeSpan(futures)).Await();
}

absl::StatusOr<std::unique_ptr<Communicator>>
OnecclCollectives::CreateCommunicator() {
  return Unimplemented("Creating a standalone oneCCL communicator is not "
                       "implemented");
}

absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
OnecclCollectives::SplitCommunicators(
    absl::Span<const Communicator* const> comms, int32_t color,
    absl::Span<const RankId> keys, const Collectives::Config& config,
    absl::Span<const DeviceRank> ranks) {
  if (keys.size() != comms.size()) {
    return InvalidArgument("Comms and keys must have the same size, got %d "
                           "and %d",
                           comms.size(), keys.size());
  }
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "SplitOnecclComms",
                               comms.size());
  std::vector<Future<std::unique_ptr<Communicator>>> futures(comms.size());
  for (size_t i = 0; i < comms.size(); ++i) {
    futures[i] = MakeFutureOn<std::unique_ptr<Communicator>>(
        *pool.AsExecutor(), [&, i]()
        -> absl::StatusOr<std::unique_ptr<Communicator>> {
      const OnecclCommunicator* parent = Cast(comms[i]);
      ccl::communicator split =
          ccl::split_communicator(parent->comm(), color, keys[i].value());
      absl::StatusOr<stream_executor::sycl::SyclExecutor*> executor =
          ToSyclExecutor(ranks[i]);
      if (!executor.ok()) {
        return executor.status();
      }
      return absl::StatusOr<std::unique_ptr<Communicator>>(
          std::make_unique<OnecclCommunicator>(*executor, std::move(split)));
    });
  }
  return JoinFutures(absl::MakeSpan(futures)).Await();
}

absl::StatusOr<void*> OnecclCollectives::Allocate(uint64_t bytes) {
  return Unimplemented("oneCCL symmetric allocation is not implemented");
}

absl::Status OnecclCollectives::Deallocate(void* location) {
  return absl::OkStatus();
}

}  // namespace xla::gpu

XLA_COLLECTIVES_REGISTER("SYCL", "oneccl", 100,
                         std::make_unique<xla::gpu::OnecclCollectives>());
