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
#include <level_zero/ze_api.h>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/call_once.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
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
#include "xla/stream_executor/sycl/sycl_gpu_runtime.h"
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
absl::Mutex ccl_env_mu;

constexpr uint32_t kIntelPciVendorId = 0x8086;

constexpr char kCclSyclAllgathervLlThreshold[] =
    "CCL_SYCL_ALLGATHERV_LL_THRESHOLD";
constexpr char kCclSyclAllreduceSimpleThreshold[] =
    "CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD";
constexpr char kCclProcessLauncher[] = "CCL_PROCESS_LAUNCHER";
constexpr char kCclLocalSize[] = "CCL_LOCAL_SIZE";
constexpr char kCclLocalRank[] = "CCL_LOCAL_RANK";
constexpr char kCclAtlTransport[] = "CCL_ATL_TRANSPORT";
constexpr char kCclAtlShm[] = "CCL_ATL_SHM";
constexpr char kFiProvider[] = "FI_PROVIDER";

absl::StatusOr<const GpuCliqueKey*> GetOnecclGpuCliqueKey(
    const CliqueKey& key) {
  auto* gpu_key = dynamic_cast<const GpuCliqueKey*>(&key);
  if (gpu_key == nullptr) {
    return InvalidArgument("oneCCL requires GPU clique keys");
  }
  return gpu_key;
}

struct OnecclEnvDefault {
  const char* name;
  const char* value;
};

struct OnecclEnvValue {
  std::string name;
  std::string value;
};

struct OnecclEnvDefaultingResult {
  std::vector<OnecclEnvDefault> defaulted;
  std::vector<OnecclEnvValue> preset;
};

bool SetOnecclEnvDefaultIfUnset(const OnecclEnvDefault& env_default,
                                OnecclEnvDefaultingResult& result) {
  const char* preset_value = std::getenv(env_default.name);
  if (preset_value != nullptr) {
    result.preset.push_back({env_default.name, preset_value});
    return false;
  }
  if (::setenv(env_default.name, env_default.value, /*overwrite=*/0) != 0) {
    LOG(WARNING) << "Failed to set oneCCL default " << env_default.name << "="
                 << env_default.value;
    return false;
  }
  result.defaulted.push_back(env_default);
  return true;
}

OnecclEnvDefaultingResult SetOnecclEnvDefaultGroupIfUnset(
    absl::Span<const OnecclEnvDefault> defaults) {
  // The process environment is global. Serialize the getenv/setenv pair so
  // racing communicator creations cannot observe a partially defaulted policy.
  // If the caller configured any variable in the group, leave the entire group
  // to the caller instead of mixing explicit values with XLA defaults.
  absl::MutexLock lock(&ccl_env_mu);
  OnecclEnvDefaultingResult result;
  for (const OnecclEnvDefault& env_default : defaults) {
    const char* preset_value = std::getenv(env_default.name);
    if (preset_value != nullptr) {
      result.preset.push_back({env_default.name, preset_value});
    }
  }
  if (!result.preset.empty()) {
    return result;
  }
  for (const OnecclEnvDefault& env_default : defaults) {
    SetOnecclEnvDefaultIfUnset(env_default, result);
  }
  return result;
}

std::string FormatOnecclEnvDefaults(
    absl::Span<const OnecclEnvDefault> values) {
  return absl::StrJoin(
      values, " ", [](std::string* out, const OnecclEnvDefault& value) {
        absl::StrAppend(out, value.name, "=", value.value);
      });
}

std::string FormatOnecclEnvValues(absl::Span<const OnecclEnvValue> values) {
  return absl::StrJoin(values, " ",
                       [](std::string* out, const OnecclEnvValue& value) {
                         absl::StrAppend(out, value.name, "=", value.value);
                       });
}

void SetAndLogOnecclEnvDefaultGroupIfUnset(
    absl::Span<const OnecclEnvDefault> defaults,
    absl::string_view description) {
  OnecclEnvDefaultingResult result =
      SetOnecclEnvDefaultGroupIfUnset(defaults);
  if (!result.defaulted.empty()) {
    VLOG(1) << "Defaulting oneCCL " << description << ": "
            << FormatOnecclEnvDefaults(result.defaulted);
  }
  if (result.defaulted.empty() && !result.preset.empty() &&
      result.preset.size() != defaults.size()) {
    LOG(WARNING) << "Not defaulting oneCCL " << description
                 << " because the environment already sets "
                 << FormatOnecclEnvValues(result.preset)
                 << "; leaving the remaining group values unset";
  }
}

bool IsIntelGpuDevice(const ::sycl::device& device) {
  try {
    if (!device.is_gpu()) return false;
    ze_device_handle_t lz_device =
        ::sycl::get_native<::sycl::backend::ext_oneapi_level_zero>(device);
    ze_device_properties_t props{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
    ze_result_t status = zeDeviceGetProperties(lz_device, &props);
    if (status != ZE_RESULT_SUCCESS) {
      VLOG(1) << "Failed to query Level Zero device properties for oneCCL "
                 "defaults: "
              << status;
      return false;
    }
    return props.type == ZE_DEVICE_TYPE_GPU &&
           props.vendorId == kIntelPciVendorId;
  } catch (const ::sycl::exception& e) {
    VLOG(1) << "Failed to query SYCL/Level Zero device info for oneCCL "
               "defaults: "
            << e.what();
    return false;
  }
}

bool AreMultiGpuIntelDevices(absl::Span<const ::sycl::device> devices) {
  if (devices.size() < 2) return false;
  return absl::c_all_of(devices, IsIntelGpuDevice);
}

bool VisibleDevicesAreMultiGpuIntelDevices() {
  absl::StatusOr<int> device_count =
      stream_executor::sycl::SyclDevicePool::GetDeviceCount();
  if (!device_count.ok()) {
    VLOG(1) << "Failed to query SYCL device count for oneCCL defaults: "
            << device_count.status();
    return false;
  }

  std::vector<::sycl::device> devices;
  devices.reserve(*device_count);
  for (int ordinal = 0; ordinal < *device_count; ++ordinal) {
    absl::StatusOr<::sycl::device> device =
        stream_executor::sycl::SyclDevicePool::GetDevice(ordinal);
    if (!device.ok()) {
      VLOG(1) << "Failed to query SYCL device " << ordinal
              << " for oneCCL defaults: " << device.status();
      return false;
    }
    devices.push_back(std::move(*device));
  }
  return AreMultiGpuIntelDevices(devices);
}

bool StreamExecutorsAreMultiGpuIntelDevices(
    absl::Span<se::StreamExecutor* const> stream_executors) {
  std::vector<::sycl::device> devices;
  devices.reserve(stream_executors.size());
  for (se::StreamExecutor* stream_executor : stream_executors) {
    auto* sycl_executor =
        dynamic_cast<stream_executor::sycl::SyclExecutor*>(stream_executor);
    if (sycl_executor == nullptr) return false;
    devices.push_back(sycl_executor->GetDevice());
  }
  return AreMultiGpuIntelDevices(devices);
}

void SetOnecclIntelGpuCollectiveEnvDefaultsIfNeeded(
    bool is_multi_gpu_intel) {
  // On tested multi-GPU Intel SYCL configurations, oneCCL's default BF16
  // allgatherv path can route through kernels whose returned events do not
  // reliably complete. Prefer conservative SYCL-kernel thresholds unless the
  // caller explicitly configures the collective environment.
  if (!is_multi_gpu_intel) return;

  static constexpr OnecclEnvDefault kIntelGpuCollectiveEnvDefaults[] = {
      {kCclSyclAllgathervLlThreshold, "1073741824"},   // 1 GiB.
      {kCclSyclAllreduceSimpleThreshold, "33554432"},  // 32 MiB.
  };
  SetAndLogOnecclEnvDefaultGroupIfUnset(
      kIntelGpuCollectiveEnvDefaults,
      "collective environment for multi-GPU Intel SYCL devices");
}

void InitOnecclOnce() {
  // ccl::init can run before communicator ranks are available, for example
  // when creating a distributed clique id. In that case, use the visible SYCL
  // device pool as a conservative fallback.
  SetOnecclIntelGpuCollectiveEnvDefaultsIfNeeded(
      VisibleDevicesAreMultiGpuIntelDevices());
  ccl::init();
}

void SetOnecclSingleProcessBootstrapEnvDefaults() {
  // Single-process GPU executions do not run under MPI or another launcher.
  // CCL_LOCAL_SIZE and CCL_LOCAL_RANK describe launcher processes, not local
  // GPU ranks, so a many-GPU all-local clique still uses one process/rank here.
  // Respect any explicit bootstrap environment already supplied by the caller.
  static constexpr OnecclEnvDefault kSingleProcessBootstrapEnvDefaults[] = {
      {kCclProcessLauncher, "none"},
      {kCclLocalSize, "1"},
      {kCclLocalRank, "0"},
      {kCclAtlTransport, "ofi"},
      {kCclAtlShm, "1"},
      {kFiProvider, "shm"},
  };
  SetAndLogOnecclEnvDefaultGroupIfUnset(
      kSingleProcessBootstrapEnvDefaults,
      "bootstrap environment for single-process oneCCL execution");
}

absl::StatusOr<CliqueIds> GetOnecclSingleProcessCliqueIds(
    const CliqueKey& key) {
  TF_ASSIGN_OR_RETURN(const GpuCliqueKey* gpu_key,
                      GetOnecclGpuCliqueKey(key));
  if (!gpu_key->is_local()) {
    return Internal(
        "single-process oneCCL topology cannot create clique ids for "
        "non-local GPU clique %v",
        key);
  }

  SetOnecclSingleProcessBootstrapEnvDefaults();
  return CliqueIds();
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
    TF_ASSIGN_OR_RETURN(const GpuCliqueKey* gpu_key,
                        GetOnecclGpuCliqueKey(key));

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

absl::StatusOr<ccl::library_version> GetOnecclLibraryVersion() {
  return OnecclValue<ccl::library_version>(
      "ccl::get_library_version", [] { return ccl::get_library_version(); });
}

std::string FormatOnecclVersion(const ccl::library_version& version) {
  std::string numeric =
      absl::StrFormat("%u.%u.%u", version.major, version.minor,
                      version.update);
  std::string formatted = numeric;
  auto append_field = [&](absl::string_view name, const char* value) {
    if (value != nullptr && value[0] != '\0') {
      absl::StrAppend(&formatted, " ", name, "=", value);
    }
  };

  append_field("status", version.product_status);
  append_field("build", version.build_date);

  std::string backend = static_cast<std::string>(version.cl_backend_name);
  if (!backend.empty()) {
    absl::StrAppend(&formatted, " backend=", backend);
  }

  if (version.full != nullptr && version.full[0] != '\0') {
    absl::string_view full(version.full);
    absl::string_view status(version.product_status != nullptr
                                 ? version.product_status
                                 : "");
    if (full != status && full != numeric) {
      absl::StrAppend(&formatted, " full=", full);
    }
  }
  return formatted;
}

std::string CompiledOnecclVersion() {
  return absl::StrFormat("%d.%d.%d", CCL_MAJOR_VERSION, CCL_MINOR_VERSION,
                         CCL_UPDATE_VERSION);
}

struct OnecclDeviceContext {
  ccl::device device;
  ccl::context context;
};

struct OnecclCommunicatorCreationSetup {
  std::vector<se::StreamExecutor*> stream_executors;
  std::shared_ptr<CancellationToken> cancel;
};

absl::StatusOr<OnecclDeviceContext> CreateOnecclDeviceContext(
    se::StreamExecutor* stream_executor) {
  auto* sycl_executor =
      dynamic_cast<stream_executor::sycl::SyclExecutor*>(stream_executor);
  if (sycl_executor == nullptr) {
    return InvalidArgument(
        "oneCCL communicator creation requires SYCL executors");
  }

  ::sycl::device sycl_device = sycl_executor->GetDevice();
  ASSIGN_OR_RETURN(ccl::device ccl_device,
                   OnecclValue<ccl::device>("ccl::create_device", [&] {
                     return ccl::create_device(std::move(sycl_device));
                   }));
  ASSIGN_OR_RETURN(::sycl::context sycl_context, sycl_executor->GetContext());
  ASSIGN_OR_RETURN(ccl::context ccl_context,
                   OnecclValue<ccl::context>("ccl::create_context", [&] {
                     return ccl::create_context(std::move(sycl_context));
                   }));
  return OnecclDeviceContext{std::move(ccl_device), std::move(ccl_context)};
}

absl::StatusOr<ccl::communicator> CreateOnecclCommunicator(
    int num_devices, RankId rank, ccl::device& device, ccl::context& context,
    const std::shared_ptr<ccl::kvs_interface>& kvs,
    const ccl::comm_attr& comm_attr) {
  return OnecclValue<ccl::communicator>("ccl::create_communicatorExt", [&] {
    return ccl::create_communicatorExt(num_devices, rank.value(), device,
                                       context, kvs, comm_attr);
  });
}

std::string KvsAddressToString(const ccl::kvs::address_type& address) {
  return std::string(address.begin(), address.end());
}

absl::StatusOr<std::unique_ptr<Communicator>> Cast(
    absl::StatusOr<std::unique_ptr<OnecclCommunicator>> comm_or) {
  ASSIGN_OR_RETURN(auto comm, std::move(comm_or));
  return std::unique_ptr<Communicator>(comm.release());
}

absl::StatusOr<std::unique_ptr<Communicator>> CreateOnecclCommunicatorForRank(
    se::StreamExecutor* stream_executor, int num_devices, RankId rank,
    const std::shared_ptr<ccl::kvs_interface>& kvs,
    const ccl::comm_attr& comm_attr,
    const std::shared_ptr<CancellationToken>& cancel) {
  int32_t device_ordinal = stream_executor->device_ordinal();
  auto activation = stream_executor->Activate();

  ASSIGN_OR_RETURN(OnecclDeviceContext device_context,
                   CreateOnecclDeviceContext(stream_executor));

  absl::StatusOr<ccl::communicator> ccl_comm = CreateOnecclCommunicator(
      num_devices, rank, device_context.device, device_context.context, kvs,
      comm_attr);
  if (!ccl_comm.ok()) {
    LOG(ERROR) << absl::StreamFormat(
        "[%d] [rank=%v] Failed to create oneCCL communicator: %s",
        device_ordinal, rank, ccl_comm.status().ToString());
    return absl::StatusOr<std::unique_ptr<Communicator>>(ccl_comm.status());
  }

  absl::StatusOr<std::unique_ptr<OnecclCommunicator>> comm =
      OnecclCommunicator::Create(stream_executor, std::move(*ccl_comm), kvs,
                                 cancel);
  if (!comm.ok()) {
    LOG(ERROR) << absl::StreamFormat(
        "[%d] [rank=%v] Failed to wrap oneCCL communicator: %s",
        device_ordinal, rank, comm.status().ToString());
  }
  return Cast(std::move(comm));
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

std::string RankIdsToString(absl::Span<const RankId> ranks) {
  return absl::StrJoin(ranks, ",", [](std::string* str, RankId rank) {
    absl::StrAppend(str, rank.value());
  });
}

absl::StatusOr<const OnecclCommunicator*> CastOneccl(const Communicator* comm) {
  if (comm == nullptr) {
    return InvalidArgument("oneCCL split communicator cannot be null");
  }
  const auto* oneccl_communicator =
      dynamic_cast<const OnecclCommunicator*>(comm);
  if (oneccl_communicator == nullptr) {
    return InvalidArgument(
        "oneCCL split requires oneCCL communicators, got a different "
        "communicator implementation");
  }
  return oneccl_communicator;
}

absl::StatusOr<std::unique_ptr<Communicator>>
CreateSplitOnecclCommunicatorForRank(
    se::StreamExecutor* stream_executor, const Communicator* comm,
    RankId rank, RankId key, int32_t color, bool split_share,
    const std::shared_ptr<CancellationToken>& cancel) {
  int32_t device_ordinal = stream_executor->device_ordinal();
  ASSIGN_OR_RETURN(const OnecclCommunicator* oneccl_comm, CastOneccl(comm));

  tsl::profiler::TraceMe trace([&] {
    return tsl::profiler::TraceMeEncode(
        absl::StrFormat("[%v] [rank=%v] onecclCommSplit", device_ordinal,
                        rank),
        {{"color", color}, {"key", key}});
  });

  absl::Time split_start = absl::Now();
  VLOG(2) << absl::StreamFormat(
      "[%d] [rank=%v] Split oneCCL communicator %p with color %d and key %v",
      device_ordinal, rank, static_cast<const void*>(comm), color, key);

  absl::StatusOr<ccl::communicator> split_comm =
      oneccl_comm->Split(color, key.value(), split_share);
  if (!split_comm.ok()) {
    LOG(ERROR) << absl::StreamFormat(
        "[%d] [rank=%v] Failed to split oneCCL communicator: %s",
        device_ordinal, rank, split_comm.status().ToString());
    return absl::StatusOr<std::unique_ptr<Communicator>>(split_comm.status());
  }

  absl::Time split_done = absl::Now();
  VLOG(2) << absl::StreamFormat(
      "[%d] [rank=%v] Split oneCCL communicator %p with color %d and key %v "
      "in %v",
      device_ordinal, rank, static_cast<const void*>(comm), color, key,
      split_done - split_start);

  absl::StatusOr<std::unique_ptr<OnecclCommunicator>> split_oneccl_comm =
      OnecclCommunicator::Create(stream_executor, std::move(*split_comm),
                                 oneccl_comm->kvs(), cancel);
  if (!split_oneccl_comm.ok()) {
    LOG(ERROR) << absl::StreamFormat(
        "[%d] [rank=%v] Failed to split oneCCL communicator: %s",
        device_ordinal, rank, split_oneccl_comm.status().ToString());
  }
  return Cast(std::move(split_oneccl_comm));
}

absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
CreateOnecclCommunicatorsForRanks(
    absl::Span<se::StreamExecutor* const> stream_executors, int num_devices,
    absl::Span<const OnecclCollectives::DeviceRank> ranks,
    const std::shared_ptr<ccl::kvs_interface>& kvs,
    const ccl::comm_attr& comm_attr,
    const std::shared_ptr<CancellationToken>& cancel) {
  if (stream_executors.size() != ranks.size()) {
    return InvalidArgument(
        "oneCCL communicator creation got %d stream executors for %d ranks",
        static_cast<int64_t>(stream_executors.size()),
        static_cast<int64_t>(ranks.size()));
  }

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "CreateOnecclComms",
                               ranks.size());

  std::vector<Future<std::unique_ptr<Communicator>>> futures(ranks.size());
  for (size_t i = 0; i < ranks.size(); ++i) {
    futures[i] = MakeFutureOn(*pool.AsExecutor(), [&, i]()
        -> absl::StatusOr<std::unique_ptr<Communicator>> {
      return CreateOnecclCommunicatorForRank(
          stream_executors[i], num_devices, ranks[i].rank, kvs, comm_attr,
          cancel);
    });
  }

  return JoinFutures(absl::MakeSpan(futures)).Await();
}

absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
SplitOnecclCommunicatorsForRanks(
    absl::Span<const Communicator* const> comms,
    absl::Span<se::StreamExecutor* const> stream_executors,
    absl::Span<const OnecclCollectives::DeviceRank> ranks, int32_t color,
    absl::Span<const RankId> keys, bool split_share,
    const std::shared_ptr<CancellationToken>& cancel) {
  if (stream_executors.size() != comms.size()) {
    return InvalidArgument(
        "oneCCL communicator split got %d stream executors for %d "
        "communicators",
        static_cast<int64_t>(stream_executors.size()),
        static_cast<int64_t>(comms.size()));
  }
  if (ranks.size() != comms.size()) {
    return InvalidArgument(
        "oneCCL communicator split got %d ranks for %d communicators",
        static_cast<int64_t>(ranks.size()), static_cast<int64_t>(comms.size()));
  }
  if (keys.size() != comms.size()) {
    return InvalidArgument(
        "oneCCL communicator split got %d keys for %d communicators",
        static_cast<int64_t>(keys.size()), static_cast<int64_t>(comms.size()));
  }

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "SplitOnecclComms",
                               comms.size());

  std::vector<Future<std::unique_ptr<Communicator>>> futures(comms.size());
  for (size_t i = 0; i < comms.size(); ++i) {
    futures[i] = MakeFutureOn(*pool.AsExecutor(), [&, i] {
      return CreateSplitOnecclCommunicatorForRank(
          stream_executors[i], comms[i], ranks[i].rank, keys[i], color,
          split_share, cancel);
    });
  }

  return JoinFutures(absl::MakeSpan(futures)).Await();
}

absl::StatusOr<ccl::comm_attr> AsOnecclCommAttr(
    const GpuCollectives::Config& config) {
  if (config.max_nchannels > 0) {
    VLOG(2) << "Maximum number of oneCCL CTAs requested, but the C++ API does "
               "not expose an equivalent communicator attribute: "
            << config.max_nchannels;
  }

  if (config.use_minimal_resource) {
    VLOG(2) << "oneCCL minimal-resource communicator mode requested, but the "
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
    auto* device = dynamic_cast<GpuCollectives::Device*>(ranks[i].device);
    if (device == nullptr) {
      return InvalidArgument(
          "oneCCL communicator creation requires GpuCollectives::Device");
    }
    stream_executors[i] = device->stream_executor();
  }
  return stream_executors;
}

absl::Status ValidateCreateCommunicatorInputs(
    const GpuCliqueKey& clique_key, const std::optional<CliqueIds>& clique_ids,
    absl::Span<const OnecclCollectives::DeviceRank> ranks) {
  if (ranks.empty()) {
    return InvalidArgument("oneCCL communicator creation requires ranks");
  }
  if (ranks.size() != clique_key.num_local_participants()) {
    return InvalidArgument(
        "oneCCL communicator creation got %d local ranks for a clique with %d "
        "local participants",
        ranks.size(), clique_key.num_local_participants());
  }
  if (clique_ids.has_value() && clique_ids->size() > 1) {
    return Unimplemented(
        "oneCCL C++ KVS bootstrap uses one address per communicator set; "
        "expected exactly one CliqueId but got %d",
        clique_ids->size());
  }
  return absl::OkStatus();
}

absl::Status ValidateCreateCommunicatorConfig(
    const GpuCollectives::Config& gpu_config) {
  if (!gpu_config.blocking_communicators && !gpu_config.async_execution) {
    return FailedPrecondition(
        "GpuCollectives::Config blocking_communicators is false, but "
        "async_execution is false. Non-blocking communicators require "
        "asynchronous execution.");
  }
  return absl::OkStatus();
}

absl::StatusOr<const GpuCollectives::Config*> GetGpuCollectivesConfig(
    const Collectives::Config& config) {
  auto* gpu_config = dynamic_cast<const GpuCollectives::Config*>(&config);
  if (gpu_config == nullptr) {
    return InvalidArgument(
        "oneCCL communicator requires GpuCollectives::Config");
  }
  return gpu_config;
}

absl::Status ValidateSplitCommunicatorInputs(
    absl::Span<const Communicator* const> comms, absl::Span<const RankId> keys,
    absl::Span<const OnecclCollectives::DeviceRank> ranks) {
  if (keys.size() != comms.size()) {
    return InvalidArgument(
        "Comms and keys must have the same size, but %d != %d", comms.size(),
        keys.size());
  }
  if (ranks.size() != comms.size()) {
    return InvalidArgument(
        "Comms and ranks must have the same size, but %d != %d", comms.size(),
        ranks.size());
  }
  return absl::OkStatus();
}

std::string FormatCliqueIdsForLog(const std::optional<CliqueIds>& clique_ids) {
  if (!clique_ids.has_value() || clique_ids->data().empty()) {
    return "no clique id";
  }
  return absl::StrFormat("size(id)=%zu; fingerprint(id)=%v",
                         clique_ids->size(), clique_ids->fingerprint());
}

absl::Status LogOnecclCommunicatorInitialization(
    const CliqueKey& clique_key, const std::optional<CliqueIds>& clique_ids,
    absl::Span<const OnecclCollectives::DeviceRank> ranks) {
  ASSIGN_OR_RETURN(ccl::library_version oneccl_version,
                   GetOnecclLibraryVersion());
  VLOG(1) << absl::StreamFormat(
      "[%s] [ranks=%s] Initialize oneCCL (compiled with %s, runtime reports "
      "%s) communicators for %d local devices (out of %d global devices); %s",
      DeviceOrdinalsToString(ranks), DeviceRanksToString(ranks),
      CompiledOnecclVersion(), FormatOnecclVersion(oneccl_version),
      ranks.size(), clique_key.num_devices(), FormatCliqueIdsForLog(clique_ids));
  return absl::OkStatus();
}

std::shared_ptr<CancellationToken> GetOrCreateCancellationToken(
    std::shared_ptr<CancellationToken> cancel) {
  if (cancel != nullptr) return cancel;
  return std::make_shared<CancellationToken>();
}

absl::Status CheckOnecclCommunicatorNotCancelled(
    const std::shared_ptr<CancellationToken>& cancel, absl::string_view action) {
  if (cancel->IsCancelled()) {
    return FailedPrecondition("oneCCL communicator %s cancelled", action);
  }
  return absl::OkStatus();
}

absl::StatusOr<OnecclCommunicatorCreationSetup>
PrepareOnecclCommunicatorCreation(
    const CliqueKey& clique_key, const std::optional<CliqueIds>& clique_ids,
    absl::Span<const OnecclCollectives::DeviceRank> ranks,
    const Collectives::Config& config,
    std::shared_ptr<CancellationToken> cancel) {
  ASSIGN_OR_RETURN(std::vector<se::StreamExecutor*> stream_executors,
                   GetStreamExecutors(ranks));
  SetOnecclIntelGpuCollectiveEnvDefaultsIfNeeded(
      StreamExecutorsAreMultiGpuIntelDevices(stream_executors));

  TF_RETURN_IF_ERROR(EnsureOnecclInitialized());
  TF_RETURN_IF_ERROR(
      LogOnecclCommunicatorInitialization(clique_key, clique_ids, ranks));

  ASSIGN_OR_RETURN(const GpuCollectives::Config* gpu_config,
                   GetGpuCollectivesConfig(config));
  TF_RETURN_IF_ERROR(ValidateCreateCommunicatorConfig(*gpu_config));

  cancel = GetOrCreateCancellationToken(std::move(cancel));
  TF_RETURN_IF_ERROR(CheckOnecclCommunicatorNotCancelled(cancel, "creation"));

  return OnecclCommunicatorCreationSetup{std::move(stream_executors),
                                         std::move(cancel)};
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

absl::StatusOr<CliqueIds> OnecclCollectives::CreateCliqueIds(
    const CliqueKey& clique_key) const {
  return GetOnecclSingleProcessCliqueIds(clique_key);
}

absl::StatusOr<std::shared_ptr<ccl::kvs_interface>>
OnecclCollectives::GetOrCreateKvs(
    const CliqueKey& clique_key, const std::optional<CliqueIds>& clique_ids,
    absl::Span<const DeviceRank> ranks, bool all_local_clique) const {
  if (all_local_clique) {
    auto kvs = std::make_shared<InMemoryOnecclKvs>();
    VLOG(2) << absl::StreamFormat(
        "[%s] [ranks=%s] Using in-memory oneCCL KVS for all-local clique; "
        "kvs_id=%d",
        DeviceOrdinalsToString(ranks), DeviceRanksToString(ranks),
        kvs->get_id());
    return kvs;
  }

  if (!clique_ids.has_value() || clique_ids->data().empty()) {
    return InvalidArgument(
        "oneCCL communicator creation requires a CliqueId for non-all-local "
        "cliques; got %d local ranks out of %d global ranks",
        ranks.size(), clique_key.num_devices());
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
      VLOG(2) << absl::StreamFormat(
          "[%s] [ranks=%s] Reusing cached oneCCL KVS; kvs_id=%d; "
          "fingerprint(id)=%v",
          DeviceOrdinalsToString(ranks), DeviceRanksToString(ranks),
          it->second->get_id(), clique_ids->fingerprint());
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
  VLOG(2) << absl::StreamFormat(
      "[%s] [ranks=%s] Created oneCCL KVS from clique id; kvs_id=%d; "
      "fingerprint(id)=%v",
      DeviceOrdinalsToString(ranks), DeviceRanksToString(ranks), kvs->get_id(),
      clique_ids->fingerprint());
  return std::static_pointer_cast<ccl::kvs_interface>(kvs);
}

absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
OnecclCollectives::CreateCommunicatorsWithCancel(
    const CliqueKey& clique_key, const std::optional<CliqueIds>& clique_ids,
    absl::Span<const DeviceRank> ranks, const Collectives::Config& config,
    std::shared_ptr<CancellationToken> cancel) {
  ASSIGN_OR_RETURN(const GpuCliqueKey* gpu_clique_key,
                   GetOnecclGpuCliqueKey(clique_key));
  TF_RETURN_IF_ERROR(
      ValidateCreateCommunicatorInputs(*gpu_clique_key, clique_ids, ranks));
  const bool all_local_clique = gpu_clique_key->is_local();
  if (all_local_clique) {
    SetOnecclSingleProcessBootstrapEnvDefaults();
  }

  ASSIGN_OR_RETURN(OnecclCommunicatorCreationSetup setup,
                   PrepareOnecclCommunicatorCreation(
                       clique_key, clique_ids, ranks, config,
                       std::move(cancel)));
  ASSIGN_OR_RETURN(std::shared_ptr<ccl::kvs_interface> kvs,
                   GetOrCreateKvs(clique_key, clique_ids, ranks,
                                  all_local_clique));
  ASSIGN_OR_RETURN(const GpuCollectives::Config* gpu_config,
                   GetGpuCollectivesConfig(config));
  ASSIGN_OR_RETURN(ccl::comm_attr comm_attr, AsOnecclCommAttr(*gpu_config));

  absl::Time init_start = absl::Now();
  tsl::profiler::TraceMe trace([&] {
    return tsl::profiler::TraceMeEncode(
        absl::StrFormat("[%s] onecclCreateCommunicator",
                        DeviceOrdinalsToString(ranks)),
        {{"num_local_ranks", ranks.size()},
         {"num_global_ranks", clique_key.num_devices()}});
  });

  ASSIGN_OR_RETURN(std::vector<std::unique_ptr<Communicator>> communicators,
                   CreateOnecclCommunicatorsForRanks(
                       setup.stream_executors,
                       static_cast<int>(clique_key.num_devices()), ranks,
                       kvs, comm_attr, setup.cancel));
  TF_RETURN_IF_ERROR(
      CheckOnecclCommunicatorNotCancelled(setup.cancel, "creation"));

  absl::Time init_done = absl::Now();
  VLOG(1) << absl::StreamFormat(
      "[%s] [ranks=%s] Initialized %d oneCCL communicators in %v",
      DeviceOrdinalsToString(ranks), DeviceRanksToString(ranks),
      communicators.size(), init_done - init_start);

  return std::move(communicators);
}

absl::StatusOr<std::vector<std::unique_ptr<Communicator>>>
OnecclCollectives::SplitCommunicatorsWithCancel(
    absl::Span<const Communicator* const> comms, int32_t color,
    absl::Span<const RankId> keys, const Collectives::Config& config,
    absl::Span<const DeviceRank> ranks,
    std::shared_ptr<CancellationToken> cancel) {
  TF_RETURN_IF_ERROR(ValidateSplitCommunicatorInputs(comms, keys, ranks));
  cancel = GetOrCreateCancellationToken(std::move(cancel));
  TF_RETURN_IF_ERROR(CheckOnecclCommunicatorNotCancelled(cancel, "split"));

  VLOG(1) << absl::StreamFormat(
      "[%s] [ranks=%s] Split %d oneCCL communicators using color %d "
      "and keys [%s]",
      DeviceOrdinalsToString(ranks), DeviceRanksToString(ranks), comms.size(),
      color, RankIdsToString(keys));

  ASSIGN_OR_RETURN(auto stream_executors, GetStreamExecutors(ranks));
  ASSIGN_OR_RETURN(const GpuCollectives::Config* gpu_config,
                   GetGpuCollectivesConfig(config));

  ASSIGN_OR_RETURN(std::vector<std::unique_ptr<Communicator>> split_comms,
                   SplitOnecclCommunicatorsForRanks(
                       comms, stream_executors, ranks, color, keys,
                       gpu_config->split_share, cancel));
  TF_RETURN_IF_ERROR(CheckOnecclCommunicatorNotCancelled(cancel, "split"));
  return split_comms;
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
    SetOnecclSingleProcessBootstrapEnvDefaults();
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
