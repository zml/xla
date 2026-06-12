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
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/base/call_once.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "oneapi/ccl.h"
#include "oneapi/ccl.hpp"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/oneccl_communicator.h"
#include "xla/backends/gpu/collectives/oneccl_errors.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/clique_key.h"
#include "xla/core/collectives/collectives_registry.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/debug_options_flags.h"
#include "xla/future.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/runtime/device_id.h"
#include "xla/runtime/process_id.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/sycl/sycl_executor.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/numbers.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::gpu {

namespace {

absl::Status OnecclExceptionStatus(absl::string_view expr,
                                   absl::string_view message);

absl::once_flag ccl_init_once;

void SetOnecclEnvDefault(const char* name, const char* value) {
  if (std::getenv(name) != nullptr) return;
  if (::setenv(name, value, /*overwrite=*/0) != 0) {
    LOG(WARNING) << "Failed to set oneCCL default " << name << "=" << value;
  }
}

void InitOnecclOnce() {
  // On Arc B-series, oneCCL's large SYCL kernels can reset the device in the
  // one-thread-per-rank configuration. Prefer oneCCL's PCIe/LL P2P path unless
  // the caller explicitly overrides these thresholds.
  SetOnecclEnvDefault("CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD", "1073741824");
  SetOnecclEnvDefault("CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD", "33554432");
  ccl::init();
}

void SetOnecclAllLocalEnvDefaults() {
  // Single-process, all-local GPU cliques do not run under MPI or another
  // launcher, so tell oneCCL to bootstrap with a single local process unless
  // the caller has explicitly configured a launcher/topology.
  SetOnecclEnvDefault("CCL_PROCESS_LAUNCHER", "none");
  SetOnecclEnvDefault("CCL_LOCAL_SIZE", "1");
  SetOnecclEnvDefault("CCL_LOCAL_RANK", "0");
  SetOnecclEnvDefault("CCL_ATL_TRANSPORT", "ofi");
}

absl::Status EnsureOnecclInitialized() {
  try {
    absl::call_once(ccl_init_once, InitOnecclOnce);
  } catch (const ccl::exception& e) {
    return OnecclExceptionStatus("ccl::init", e.what());
  } catch (const std::exception& e) {
    return OnecclExceptionStatus("ccl::init", e.what());
  } catch (...) {
    return OnecclExceptionStatus("ccl::init", "unknown exception");
  }
  return absl::OkStatus();
}

// OnecclIdStore generates clique unique ids for GPU cliques using oneCCL APIs.
// The unique id is exchanged through the process-wide KV store so all ranks in
// a distributed clique can bootstrap the same oneCCL communicator.
class OnecclIdStore {
 public:
  OnecclIdStore(
      ProcessId process_id,
      absl::flat_hash_map<GlobalDeviceId, ProcessId> device_to_process,
      std::shared_ptr<KeyValueStoreInterface> kv_store)
      : process_id_(process_id),
        device_to_process_(std::move(device_to_process)),
        kv_store_(std::move(kv_store)) {}

  absl::StatusOr<CliqueIds> GetCliqueIds(
      const CliqueKey& key, OnecclCollectives& oneccl_collectives) {
    auto* gpu_key = absl::down_cast<const GpuCliqueKey*>(&key);
    if (gpu_key == nullptr) {
      return InvalidArgument("Expected GPU clique key");
    }

    {
      absl::MutexLock lock(mu_);
      auto it = cache_.find(*gpu_key);
      if (it != cache_.end()) {
        return it->second;
      }
    }

    // oneCCL exposes a single unique id based communicator initialization API.
    int64_t nroots = 1;

    auto kv_key = [&](ProcessId root_process) {
      return absl::StrFormat(
          "oneccl root_process: %v; clique: devices=[%s]; "
          "communication_id=%v; incarnations=[%s]",
          root_process, absl::StrJoin(key.devices(), ","),
          gpu_key->communication_id(),
          absl::StrJoin(gpu_key->incarnations(), ",",
                        [](std::string* out, IncarnationId id) {
                          absl::StrAppend(out, id.value());
                        }));
    };

    std::vector<GlobalDeviceId> root_devices = gpu_key->GetRootDevices(nroots);
    absl::btree_set<ProcessId> root_processes;
    for (GlobalDeviceId root : root_devices) {
      root_processes.insert(device_to_process_.at(root));
    }

    VLOG(4) << absl::StreamFormat(
        "Get oneCCL clique ids: process=%v; root_devices=%d:[%s]; "
        "root_processes=%d:[%s]; clique=%v",
        process_id_, root_devices.size(), HumanReadableDevices(root_devices),
        root_processes.size(),
        HumanReadableProcesses(std::vector<ProcessId>(root_processes.begin(),
                                                      root_processes.end())),
        key);

    if (root_processes.contains(process_id_)) {
      absl::Time set_clique_id_start = absl::Now();
      ASSIGN_OR_RETURN(CliqueId clique_id,
                       oneccl_collectives.CreateUniqueCliqueId());
      RETURN_IF_ERROR(
          kv_store_->Set(kv_key(process_id_), clique_id.ToString()));
      absl::Time set_clique_id_done = absl::Now();
      VLOG(5) << absl::StreamFormat("Set oneCCL clique id process=%v in %v",
                                    process_id_,
                                    set_clique_id_done - set_clique_id_start);
    }

    absl::Time get_clique_ids_start = absl::Now();
    CliqueIds clique_ids;
    for (ProcessId root : root_processes) {
      ASSIGN_OR_RETURN(std::string id_str,
                       kv_store_->Get(kv_key(root), absl::Minutes(10)));
      clique_ids.Add(CliqueId(id_str));
    }
    absl::Time get_clique_ids_done = absl::Now();

    VLOG(5) << absl::StreamFormat(
        "Got oneCCL clique ids in %v: root_devices=%d:[%s]; "
        "root_processes=%d:[%s]; clique=%v",
        get_clique_ids_done - get_clique_ids_start, root_devices.size(),
        HumanReadableDevices(root_devices), root_processes.size(),
        HumanReadableProcesses(std::vector<ProcessId>(root_processes.begin(),
                                                      root_processes.end())),
        key);

    absl::MutexLock lock(mu_);
    auto result = cache_.emplace(*gpu_key, std::move(clique_ids));
    TF_RET_CHECK(result.second) << "Clique IDs already in cache";
    return result.first->second;
  }

 private:
  ProcessId process_id_;
  absl::flat_hash_map<GlobalDeviceId, ProcessId> device_to_process_;
  std::shared_ptr<KeyValueStoreInterface> kv_store_;

  absl::Mutex mu_;
  absl::flat_hash_map<GpuCliqueKey, CliqueIds> cache_ ABSL_GUARDED_BY(mu_);
};

class InMemoryOnecclKvs final : public ccl::kvs_interface {
 public:
  ccl::vector_class<char> get(const ccl::string_class& key) override {
    const std::string owned_key(key);
    absl::MutexLock lock(mu_);
    while (!entries_.contains(owned_key)) {
      cond_.Wait(&mu_);
    }
    return entries_.at(owned_key);
  }

  void set(const ccl::string_class& key,
           const ccl::vector_class<char>& data) override {
    {
      absl::MutexLock lock(mu_);
      entries_[std::string(key)] = data;
    }
    cond_.SignalAll();
  }

  int get_id() override { return id_; }

 private:
  static int NextId() {
    static std::atomic<int> next_id{1};
    return next_id.fetch_add(1);
  }

  const int id_ = NextId();
  absl::Mutex mu_;
  absl::CondVar cond_;
  absl::flat_hash_map<std::string, ccl::vector_class<char>> entries_
      ABSL_GUARDED_BY(mu_);
};

absl::Status OnecclExceptionStatus(absl::string_view expr,
                                   absl::string_view message) {
  return absl::InternalError(
      absl::StrFormat("oneCCL call failed: %s: %s", expr, message));
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

absl::StatusOr<ccl::library_version> GetLinkedOnecclVersion() {
  return OnecclValue<ccl::library_version>(
      "ccl::get_library_version", [] { return ccl::get_library_version(); });
}

std::string FormatOnecclVersion(const ccl::library_version& version) {
  if (version.full != nullptr) {
    return version.full;
  }
  return absl::StrFormat("%d.%d.%d", version.major, version.minor,
                         version.update);
}

std::string CompiledOnecclVersion() {
  return absl::StrFormat("%d.%d.%d", CCL_MAJOR_VERSION, CCL_MINOR_VERSION,
                         CCL_UPDATE_VERSION);
}

std::string KvsAddressToString(const ccl::kvs::address_type& address) {
  return std::string(address.begin(), address.end());
}

absl::StatusOr<std::unique_ptr<Communicator>> Cast(
    absl::StatusOr<std::unique_ptr<OnecclCommunicator>> comm_or) {
  ASSIGN_OR_RETURN(auto comm, std::move(comm_or));
  return std::unique_ptr<Communicator>(comm.release());
}

int32_t DeviceOrdinal(const OnecclCollectives::DeviceRank& rank) {
  auto* device = absl::down_cast<const GpuCollectives::Device*>(rank.device);
  return device->stream_executor()->device_ordinal();
}

std::string DeviceOrdinalsToString(
    absl::Span<const OnecclCollectives::DeviceRank> ranks) {
  return absl::StrJoin(ranks, ",", [](std::string* str, auto& rank) {
    auto* device = tsl::down_cast<const GpuCollectives::Device*>(rank.device);
    absl::StrAppend(str, device->stream_executor()->device_ordinal());
  });
}

std::string DeviceRanksToString(
    absl::Span<const OnecclCollectives::DeviceRank> ranks) {
  return absl::StrJoin(ranks, ",", [](std::string* str, auto& rank) {
    absl::StrAppend(str, rank.rank.value());
  });
}

const OnecclCommunicator* CastOneccl(const Communicator* comm) {
  auto* oneccl_communicator =
      absl::down_cast<const OnecclCommunicator*>(comm);
  CHECK(oneccl_communicator != nullptr) << "Unsupported XLA communicator";
  return oneccl_communicator;
}

absl::StatusOr<ccl::comm_attr> AsOnecclCommAttr(
    const GpuCollectives::Config& config) {
  if (config.max_nchannels > 0) {
    VLOG(1) << "Maximum number of oneCCL CTAs requested, but the C++ API does "
               "not expose an equivalent communicator attribute: "
            << config.max_nchannels;
  }

  if (config.use_minimal_resource) {
    VLOG(1) << "oneCCL minimal-resource communicator mode requested, but the "
               "C++ API does not expose an equivalent control.";
  }

  return OnecclValue<ccl::comm_attr>("ccl::create_comm_attr", [&] {
    return ccl::create_comm_attr(
        ccl::attr_val<ccl::comm_attr_id::blocking>(
            config.blocking_communicators ? 1 : 0));
  });
}

absl::StatusOr<std::vector<se::StreamExecutor*>> GetStreamExecutors(
    absl::Span<const OnecclCollectives::DeviceRank> ranks) {
  std::vector<se::StreamExecutor*> stream_executors(ranks.size());
  for (size_t i = 0; i < ranks.size(); ++i) {
    auto* device = absl::down_cast<GpuCollectives::Device*>(ranks[i].device);
    TF_RET_CHECK(device) << "Device must be GpuCollectives::Device";
    stream_executors[i] = device->stream_executor();
  }
  return stream_executors;
}

}  // namespace

absl::StatusOr<CliqueId> OnecclCollectives::CreateUniqueCliqueId() const {
  TF_RETURN_IF_ERROR(EnsureOnecclInitialized());

  ASSIGN_OR_RETURN(std::shared_ptr<ccl::kvs> kvs,
                   OnecclValue<std::shared_ptr<ccl::kvs>>(
                       "ccl::create_main_kvs",
                       [] { return ccl::create_main_kvs(); }));
  std::string id = KvsAddressToString(kvs->get_address());

  {
    absl::MutexLock lock(mu_);
    kvs_cache_[id] = kvs;
  }

  return CliqueId(absl::string_view(id));
}

absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
OnecclCollectives::CreateCommunicatorsWithCancel(
    const CliqueKey& clique_key, const std::optional<CliqueIds>& clique_ids,
    absl::Span<const DeviceRank> ranks, const Collectives::Config& config,
    std::shared_ptr<CancellationToken> cancel) {
  if (ranks.empty()) {
    return InvalidArgument("oneCCL communicator creation requires ranks");
  }
  if (ranks.size() == clique_key.num_devices()) {
    SetOnecclAllLocalEnvDefaults();
  }

  TF_RETURN_IF_ERROR(EnsureOnecclInitialized());

  if (clique_ids.has_value() && clique_ids->size() > 1) {
    return Unimplemented(
        "oneCCL C++ KVS bootstrap uses one address per communicator set; "
        "expected exactly one CliqueId but got %d",
        clique_ids->size());
  }

  ASSIGN_OR_RETURN(ccl::library_version oneccl_version,
                   GetLinkedOnecclVersion());
  std::string clique_id_log =
      clique_ids.has_value()
          ? absl::StrFormat("size(id)=%zu; fingerprint(id)=%v",
                            clique_ids->size(), clique_ids->fingerprint())
          : "no clique id";
  VLOG(1) << absl::StreamFormat(
      "[%s] [ranks=%s] Initialize oneCCL (compiled with %s, linked with %s) "
      "communicators for %d local devices (out of %d global devices); %s",
      DeviceOrdinalsToString(ranks), DeviceRanksToString(ranks),
      CompiledOnecclVersion(), FormatOnecclVersion(oneccl_version),
      ranks.size(), clique_key.num_devices(), clique_id_log);

  const auto& gpu_config =
      absl::down_cast<const GpuCollectives::Config&>(config);
  if (!gpu_config.blocking_communicators && !gpu_config.async_execution) {
    return FailedPrecondition(
        "GpuCollectives::Config blocking_communicators is false, but "
        "async_execution is false. Non-blocking communicators require "
        "asynchronous execution.");
  }
  if (cancel == nullptr) {
    cancel = std::make_shared<CancellationToken>();
  }
  if (cancel->IsCancelled()) {
    return FailedPrecondition("oneCCL communicator creation cancelled");
  }

  ASSIGN_OR_RETURN(auto stream_executors, GetStreamExecutors(ranks));

  auto get_or_create_kvs =
      [&]() -> absl::StatusOr<std::shared_ptr<ccl::kvs_interface>> {
    if (ranks.size() == clique_key.num_devices()) {
      return std::make_shared<InMemoryOnecclKvs>();
    }

    if (!clique_ids.has_value() || clique_ids->data().empty()) {
      return std::make_shared<InMemoryOnecclKvs>();
    }

    const CliqueId& clique_id = clique_ids->at(0);
    if (clique_id.size() != ccl::kvs::address_max_size) {
      return Internal("oneCCL KVS CliqueId size mismatch: %d vs %d",
                      clique_id.size(), ccl::kvs::address_max_size);
    }

    std::string id(clique_id.data().begin(), clique_id.data().end());
    {
      absl::MutexLock lock(mu_);
      auto it = kvs_cache_.find(id);
      if (it != kvs_cache_.end()) {
        return std::static_pointer_cast<ccl::kvs_interface>(it->second);
      }
    }

    ccl::kvs::address_type address;
    std::copy(clique_id.data().begin(), clique_id.data().end(),
              address.begin());
    ASSIGN_OR_RETURN(std::shared_ptr<ccl::kvs> kvs,
                     OnecclValue<std::shared_ptr<ccl::kvs>>(
                         "ccl::create_kvs",
                         [&] { return ccl::create_kvs(address); }));
    return std::static_pointer_cast<ccl::kvs_interface>(kvs);
  };

  ASSIGN_OR_RETURN(std::shared_ptr<ccl::kvs_interface> kvs,
                   get_or_create_kvs());
  ASSIGN_OR_RETURN(ccl::comm_attr comm_attr, AsOnecclCommAttr(gpu_config));

  absl::Time init_start = absl::Now();
  tsl::profiler::TraceMe trace([&] {
    return tsl::profiler::TraceMeEncode(
        absl::StrFormat("[%s] onecclCreateCommunicator",
                        DeviceOrdinalsToString(ranks)),
        {{"num_local_ranks", ranks.size()},
         {"num_global_ranks", clique_key.num_devices()}});
  });

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "CreateOnecclComms",
                               ranks.size());

  std::vector<Future<std::unique_ptr<Communicator>>> futures(ranks.size());
  for (size_t i = 0; i < ranks.size(); ++i) {
    futures[i] = MakeFutureOn(*pool.AsExecutor(), [&, i]()
        -> absl::StatusOr<std::unique_ptr<Communicator>> {
      int32_t device_ordinal = DeviceOrdinal(ranks[i]);
      RankId rank = ranks[i].rank;
      auto activation = stream_executors[i]->Activate();

      auto* sycl_executor =
          dynamic_cast<stream_executor::sycl::SyclExecutor*>(
              stream_executors[i]);
      TF_RET_CHECK(sycl_executor != nullptr)
          << "oneCCL communicator creation requires SYCL executors";

      ::sycl::device sycl_device = sycl_executor->GetDevice();
      ASSIGN_OR_RETURN(ccl::device ccl_device,
                       OnecclValue<ccl::device>("ccl::create_device", [&] {
                         return ccl::create_device(std::move(sycl_device));
                       }));
      ASSIGN_OR_RETURN(::sycl::context sycl_context,
                       sycl_executor->GetContext());
      ASSIGN_OR_RETURN(ccl::context ccl_context,
                       OnecclValue<ccl::context>("ccl::create_context", [&] {
                         return ccl::create_context(std::move(sycl_context));
                       }));

      VLOG(1) << absl::StreamFormat(
          "[%d] [rank=%v] Created oneCCL device/context for global size %d",
          device_ordinal, rank, clique_key.num_devices());

      absl::StatusOr<ccl::communicator> ccl_comm =
          OnecclValue<ccl::communicator>("ccl::create_communicatorExt", [&] {
            return ccl::create_communicatorExt(
                static_cast<int>(clique_key.num_devices()), rank.value(),
                ccl_device, ccl_context, kvs, comm_attr);
          });
      if (!ccl_comm.ok()) {
        LOG(ERROR) << absl::StreamFormat(
            "[%d] [rank=%v] Failed to create oneCCL communicator: %s",
            device_ordinal, rank, ccl_comm.status().ToString());
        return absl::StatusOr<std::unique_ptr<Communicator>>(
            ccl_comm.status());
      }

      absl::StatusOr<std::unique_ptr<OnecclCommunicator>> comm =
          OnecclCommunicator::Create(stream_executors[i], std::move(*ccl_comm),
                                     kvs, cancel);
      if (!comm.ok()) {
        LOG(ERROR) << absl::StreamFormat(
            "[%d] [rank=%v] Failed to wrap oneCCL communicator: %s",
            device_ordinal, rank, comm.status().ToString());
      }
      return Cast(std::move(comm));
    });
  }

  ASSIGN_OR_RETURN(std::vector<std::unique_ptr<Communicator>> communicators,
                   JoinFutures(absl::MakeSpan(futures)).Await());
  if (cancel->IsCancelled()) {
    return FailedPrecondition("oneCCL communicator creation cancelled");
  }

  absl::Time init_done = absl::Now();
  VLOG(1) << absl::StreamFormat(
      "[%s] [ranks=%s] Initialized %d oneCCL communicators in %v",
      DeviceOrdinalsToString(ranks), DeviceRanksToString(ranks),
      communicators.size(), init_done - init_start);

  return communicators;
}

absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
OnecclCollectives::SplitCommunicatorsWithCancel(
    absl::Span<const Communicator* const> comms, int32_t color,
    absl::Span<const RankId> keys, const Collectives::Config& config,
    absl::Span<const DeviceRank> ranks,
    std::shared_ptr<CancellationToken> cancel) {
  auto rank_formatter = [](std::string* str, RankId rank) {
    absl::StrAppend(str, rank.value());
  };

  VLOG(1) << absl::StreamFormat(
      "[%s] [ranks=%s] Split %d oneCCL communicators using color %d "
      "and keys [%s]",
      DeviceOrdinalsToString(ranks), DeviceRanksToString(ranks), comms.size(),
      color, absl::StrJoin(keys, ",", rank_formatter));

  if (keys.size() != comms.size()) {
    return InvalidArgument(
        "Comms and keys must have the same size, but %d != %d", comms.size(),
        keys.size());
  }

  ASSIGN_OR_RETURN(auto stream_executors, GetStreamExecutors(ranks));
  const auto& gpu_config =
      absl::down_cast<const GpuCollectives::Config&>(config);

  auto make_comm = [&](int i) -> absl::StatusOr<ccl::communicator> {
    int32_t device_ordinal = DeviceOrdinal(ranks[i]);
    RankId rank = ranks[i].rank;
    RankId key = keys[i];

    tsl::profiler::TraceMe trace([&] {
      return tsl::profiler::TraceMeEncode(
          absl::StrFormat("[%v] [rank=%v] onecclCommSplit", device_ordinal,
                          rank),
          {{"color", color}, {"key", key}});
    });

    absl::Time split_start = absl::Now();
    VLOG(1) << absl::StreamFormat(
        "[%d] [rank=%v] Split oneCCL communicator %p with color %d "
        "and key %v",
        device_ordinal, rank, static_cast<const void*>(comms[i]), color, key);

    ASSIGN_OR_RETURN(ccl::communicator split_comm,
                     CastOneccl(comms[i])->Split(color, key.value(),
                                                 gpu_config.split_share));

    absl::Time split_done = absl::Now();
    VLOG(1) << absl::StreamFormat(
        "[%d] [rank=%v] Split oneCCL communicator %p with color %d "
        "and key %v in %v",
        device_ordinal, rank, static_cast<const void*>(comms[i]), color, key,
        split_done - split_start);

    return split_comm;
  };

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "SplitOnecclComms",
                               comms.size());

  std::vector<Future<std::unique_ptr<Communicator>>> futures(comms.size());
  for (size_t i = 0; i < comms.size(); ++i) {
    futures[i] = MakeFutureOn(*pool.AsExecutor(), [&, i] {
      absl::StatusOr<ccl::communicator> split_comm = make_comm(i);
      if (!split_comm.ok()) {
        LOG(ERROR) << absl::StreamFormat(
            "[%d] [rank=%v] Failed to split oneCCL communicator: %s",
            DeviceOrdinal(ranks[i]), ranks[i].rank,
            split_comm.status().ToString());
        return absl::StatusOr<std::unique_ptr<Communicator>>(
            split_comm.status());
      }

      absl::StatusOr<std::unique_ptr<OnecclCommunicator>> comm =
          OnecclCommunicator::Create(
              stream_executors[i], std::move(*split_comm),
              CastOneccl(comms[i])->kvs(), cancel);
      if (!comm.ok()) {
        LOG(ERROR) << absl::StreamFormat(
            "[%d] [rank=%v] Failed to split oneCCL communicator: %s",
            DeviceOrdinal(ranks[i]), ranks[i].rank, comm.status().ToString());
      }
      return Cast(std::move(comm));
    });
  }

  return JoinFutures(absl::MakeSpan(futures)).Await();
}

absl::StatusOr<std::unique_ptr<Communicator>>
OnecclCollectives::CreateCommunicator() {
  return Unimplemented(
      "oneCCL communicator creation requires a clique id and rank");
}

absl::StatusOr<void*> OnecclCollectives::Allocate(uint64_t bytes) {
  void* ptr = nullptr;
  XLA_ONECCL_RETURN_IF_ERROR(onecclMemAlloc(&ptr, bytes));
  return ptr;
}

absl::Status OnecclCollectives::Deallocate(void* location) {
  return XLA_ONECCL_STATUS(onecclMemFree(location));
}

absl::StatusOr<GpuCollectives::CliqueIdCallback>
OnecclCollectives::InitializeTopology(const Topology& topology) {
  if (topology.num_processes == 1) {
    SetOnecclAllLocalEnvDefaults();
    return nullptr;
  }

  if (topology.num_processes > 1) {
    auto oneccl_id_store = std::make_shared<OnecclIdStore>(
        topology.process_id, topology.device_to_process,
        std::move(topology.kv_store));
    return [oneccl_id_store, this](const CliqueKey& key) {
      return oneccl_id_store->GetCliqueIds(key, *this);
    };
  }

  return nullptr;
}

}  // namespace xla::gpu

XLA_COLLECTIVES_REGISTER("SYCL", "oneccl", 100,
                         std::make_unique<xla::gpu::OnecclCollectives>());
