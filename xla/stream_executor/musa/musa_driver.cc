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

#include "xla/stream_executor/musa/musa_driver.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "musa.h"
#include "xla/stream_executor/musa/musa_dso_loader.h"
#include "xla/stream_executor/musa/musa_status.h"

namespace stream_executor::musa {
namespace {

// Do not use the PFN_mu* aliases from musaTypedefs.h. In the 4.0.1 SDK some
// aliases describe newer signatures than the DSO actually exports. These
// declarations are the exact ABI of each logical operation used below.
using MuInitFn = MUresult(MUSAAPI*)(unsigned int flags);
using MuGetErrorNameFn = MUresult(MUSAAPI*)(MUresult error, const char** name);
using MuGetErrorStringFn = MUresult(MUSAAPI*)(MUresult error,
                                              const char** description);
using MuGetProcAddressFn = MUresult(MUSAAPI*)(const char* symbol,
                                              void** address, int musa_version,
                                              muuint64_t flags);

using MuDriverGetVersionFn = MUresult(MUSAAPI*)(int* version);
using MuDeviceGetCountFn = MUresult(MUSAAPI*)(int* count);
using MuDeviceGetFn = MUresult(MUSAAPI*)(MUdevice* device, int ordinal);
using MuDeviceCanAccessPeerFn = MUresult(MUSAAPI*)(int* can_access_peer,
                                                   MUdevice source,
                                                   MUdevice peer);
using MuDeviceGetP2PAttributeFn =
    MUresult(MUSAAPI*)(int* value, MUdevice_P2PAttribute attribute,
                       MUdevice source, MUdevice peer);
using MuDevicePrimaryCtxRetainFn = MUresult(MUSAAPI*)(MUcontext* context,
                                                      MUdevice device);
using MuDevicePrimaryCtxReleaseV2Fn = MUresult(MUSAAPI*)(MUdevice device);
using MuDevicePrimaryCtxGetStateFn = MUresult(MUSAAPI*)(MUdevice device,
                                                        unsigned int* flags,
                                                        int* active);
using MuDevicePrimaryCtxSetFlagsV2Fn = MUresult(MUSAAPI*)(MUdevice device,
                                                          unsigned int flags);
using MuCtxSetCurrentFn = MUresult(MUSAAPI*)(MUcontext context);
using MuCtxGetCurrentFn = MUresult(MUSAAPI*)(MUcontext* context);
using MuCtxGetDeviceFn = MUresult(MUSAAPI*)(MUdevice* device);
using MuCtxSynchronizeFn = MUresult(MUSAAPI*)();
using MuCtxEnablePeerAccessFn = MUresult(MUSAAPI*)(MUcontext peer_context,
                                                   unsigned int flags);
using MuPointerGetAttributeFn = MUresult(MUSAAPI*)(
    void* value, MUpointer_attribute attribute, MUdeviceptr pointer);
using MuMemcpyPeerAsyncFn = MUresult(MUSAAPI*)(MUdeviceptr destination,
                                               MUcontext destination_context,
                                               MUdeviceptr source,
                                               MUcontext source_context,
                                               size_t bytes, MUstream stream);
using MuModuleLoadDataFn = MUresult(MUSAAPI*)(MUmodule* module,
                                              const void* image);
using MuModuleUnloadFn = MUresult(MUSAAPI*)(MUmodule module);
using MuModuleGetFunctionFn = MUresult(MUSAAPI*)(MUfunction* function,
                                                 MUmodule module,
                                                 const char* name);
using MuModuleGetGlobalV2Fn = MUresult(MUSAAPI*)(MUdeviceptr* address,
                                                 size_t* size, MUmodule module,
                                                 const char* name);
using MuFuncGetAttributeFn = MUresult(MUSAAPI*)(int* value,
                                                MUfunction_attribute attribute,
                                                MUfunction function);
using MuOccupancyMaxActiveBlocksPerMultiprocessorFn =
    MUresult(MUSAAPI*)(int* blocks, MUfunction function, int block_size,
                       size_t dynamic_shared_memory_bytes);
using MuLaunchKernelFn = MUresult(MUSAAPI*)(
    MUfunction function, unsigned int grid_dim_x, unsigned int grid_dim_y,
    unsigned int grid_dim_z, unsigned int block_dim_x, unsigned int block_dim_y,
    unsigned int block_dim_z, unsigned int shared_memory_bytes, MUstream stream,
    void** kernel_parameters, void** extra);
using MuMemsetD32AsyncFn = MUresult(MUSAAPI*)(MUdeviceptr destination,
                                              unsigned int value, size_t count,
                                              MUstream stream);
using MuGraphCreateFn = MUresult(MUSAAPI*)(MUgraph*, unsigned int);
using MuGraphDestroyFn = MUresult(MUSAAPI*)(MUgraph);
using MuGraphAddEmptyNodeFn = MUresult(MUSAAPI*)(MUgraphNode*, MUgraph,
                                                 const MUgraphNode*, size_t);
using MuGraphAddKernelNodeFn =
    MUresult(MUSAAPI*)(MUgraphNode*, MUgraph, const MUgraphNode*, size_t,
                       const MUSA_KERNEL_NODE_PARAMS*);
using MuGraphKernelNodeSetParamsFn =
    MUresult(MUSAAPI*)(MUgraphNode, const MUSA_KERNEL_NODE_PARAMS*);
using MuGraphAddMemcpyNodeFn = MUresult(MUSAAPI*)(MUgraphNode*, MUgraph,
                                                  const MUgraphNode*, size_t,
                                                  const MUSA_MEMCPY3D*,
                                                  MUcontext);
using MuGraphMemcpyNodeSetParamsFn = MUresult(MUSAAPI*)(MUgraphNode,
                                                        const MUSA_MEMCPY3D*);
using MuGraphAddMemsetNodeFn =
    MUresult(MUSAAPI*)(MUgraphNode*, MUgraph, const MUgraphNode*, size_t,
                       const MUSA_MEMSET_NODE_PARAMS*, MUcontext);
using MuGraphMemsetNodeSetParamsFn =
    MUresult(MUSAAPI*)(MUgraphNode, const MUSA_MEMSET_NODE_PARAMS*);
using MuGraphGetNodesFn = MUresult(MUSAAPI*)(MUgraph, MUgraphNode*, size_t*);
using MuGraphInstantiateFn = MUresult(MUSAAPI*)(MUgraphExec*, MUgraph,
                                                MUgraphNode*, char*, size_t);
using MuGraphExecDestroyFn = MUresult(MUSAAPI*)(MUgraphExec);
using MuGraphLaunchFn = MUresult(MUSAAPI*)(MUgraphExec, MUstream);
using MuStreamSynchronizeFn = MUresult(MUSAAPI*)(MUstream);
using MuStreamBeginCaptureFn = MUresult(MUSAAPI*)(MUstream,
                                                  MUstreamCaptureMode);
using MuStreamEndCaptureFn = MUresult(MUSAAPI*)(MUstream, MUgraph*);

constexpr muuint64_t kProcAddressFlags =
    static_cast<muuint64_t>(MU_GET_PROC_ADDRESS_LEGACY_STREAM);

struct BootstrapApi {
  MuInitFn init = nullptr;
  MuGetErrorNameFn get_error_name = nullptr;
  MuGetErrorStringFn get_error_string = nullptr;
  MuGetProcAddressFn get_proc_address = nullptr;
};

template <typename Fn>
absl::StatusOr<Fn> ResolveDirect(internal::MusaSymbolLoader& loader,
                                 absl::string_view symbol) {
  absl::StatusOr<void*> address = loader.Resolve(symbol);
  if (!address.ok()) return address.status();
  return reinterpret_cast<Fn>(*address);
}

template <typename Fn>
absl::StatusOr<Fn> ResolveRequired(internal::MusaSymbolLoader& loader,
                                   const BootstrapApi& bootstrap,
                                   absl::string_view logical_name,
                                   std::vector<std::string> dlsym_aliases) {
  std::vector<std::string> attempts;
  void* address = nullptr;

  if (bootstrap.get_proc_address != nullptr) {
    MUresult result =
        bootstrap.get_proc_address(std::string(logical_name).c_str(), &address,
                                   MUSA_VERSION, kProcAddressFlags);
    attempts.push_back(absl::StrCat(
        "muGetProcAddress(result=", static_cast<int>(result),
        ", address=", address == nullptr ? "null" : "non-null", ")"));
    if (result == MUSA_SUCCESS && address != nullptr) {
      return reinterpret_cast<Fn>(address);
    }
  } else {
    attempts.push_back("muGetProcAddress(unavailable)");
  }

  for (const std::string& alias : dlsym_aliases) {
    absl::StatusOr<void*> resolved = loader.Resolve(alias);
    if (resolved.ok() && *resolved != nullptr) {
      return reinterpret_cast<Fn>(*resolved);
    }
    attempts.push_back(absl::StrCat(
        "dlsym(", alias,
        ")=", resolved.ok() ? "null" : resolved.status().message()));
  }

  return absl::NotFoundError(absl::StrCat(
      "Failed to resolve required MUSA driver API ", logical_name,
      " with MUSA_VERSION=", MUSA_VERSION, ", flags=", kProcAddressFlags,
      "; attempts: ", absl::StrJoin(attempts, "; "), "; aliases=[",
      absl::StrJoin(dlsym_aliases, ", "), "]; DSO=", loader.loaded_path()));
}

template <typename Fn>
Fn ResolveOptional(internal::MusaSymbolLoader& loader,
                   const BootstrapApi& bootstrap,
                   absl::string_view logical_name,
                   std::vector<std::string> dlsym_aliases) {
  absl::StatusOr<Fn> resolved = ResolveRequired<Fn>(
      loader, bootstrap, logical_name, std::move(dlsym_aliases));
  if (!resolved.ok()) return nullptr;
  return *resolved;
}

}  // namespace

struct MusaDriver::Api {
  MuGetErrorNameFn get_error_name = nullptr;
  MuGetErrorStringFn get_error_string = nullptr;
  MuDriverGetVersionFn driver_get_version = nullptr;
  MuDeviceGetCountFn device_get_count = nullptr;
  MuDeviceGetFn device_get = nullptr;
  MuDeviceCanAccessPeerFn device_can_access_peer = nullptr;
  MuDeviceGetP2PAttributeFn device_get_p2p_attribute = nullptr;
  MuDevicePrimaryCtxRetainFn primary_context_retain = nullptr;
  MuDevicePrimaryCtxReleaseV2Fn primary_context_release = nullptr;
  MuDevicePrimaryCtxGetStateFn primary_context_get_state = nullptr;
  MuDevicePrimaryCtxSetFlagsV2Fn primary_context_set_flags = nullptr;
  MuCtxSetCurrentFn context_set_current = nullptr;
  MuCtxGetCurrentFn context_get_current = nullptr;
  MuCtxGetDeviceFn context_get_device = nullptr;
  MuCtxSynchronizeFn context_synchronize = nullptr;
  MuCtxEnablePeerAccessFn context_enable_peer_access = nullptr;
  MuPointerGetAttributeFn pointer_get_attribute = nullptr;
  MuMemcpyPeerAsyncFn memcpy_peer_async = nullptr;
  MuModuleLoadDataFn module_load_data = nullptr;
  MuModuleUnloadFn module_unload = nullptr;
  MuModuleGetFunctionFn module_get_function = nullptr;
  MuModuleGetGlobalV2Fn module_get_global = nullptr;
  MuFuncGetAttributeFn function_get_attribute = nullptr;
  MuOccupancyMaxActiveBlocksPerMultiprocessorFn occupancy_max_active_blocks =
      nullptr;
  MuLaunchKernelFn launch_kernel = nullptr;
  MuMemsetD32AsyncFn memset_d32_async = nullptr;
  MuGraphCreateFn graph_create = nullptr;
  MuGraphDestroyFn graph_destroy = nullptr;
  MuGraphAddEmptyNodeFn graph_add_empty_node = nullptr;
  MuGraphAddKernelNodeFn graph_add_kernel_node = nullptr;
  MuGraphKernelNodeSetParamsFn graph_kernel_node_set_params = nullptr;
  MuGraphAddMemcpyNodeFn graph_add_memcpy_node = nullptr;
  MuGraphMemcpyNodeSetParamsFn graph_memcpy_node_set_params = nullptr;
  MuGraphAddMemsetNodeFn graph_add_memset_node = nullptr;
  MuGraphMemsetNodeSetParamsFn graph_memset_node_set_params = nullptr;
  MuGraphGetNodesFn graph_get_nodes = nullptr;
  MuGraphGetNodesFn graph_get_root_nodes = nullptr;
  MuGraphInstantiateFn graph_instantiate = nullptr;
  MuGraphExecDestroyFn graph_exec_destroy = nullptr;
  MuGraphLaunchFn graph_launch = nullptr;
  MuStreamSynchronizeFn stream_synchronize = nullptr;
  MuStreamBeginCaptureFn stream_begin_capture = nullptr;
  MuStreamEndCaptureFn stream_end_capture = nullptr;
};

MusaDriver::MusaDriver() : MusaDriver(internal::CreateMusaDriverDsoLoader()) {}

MusaDriver::MusaDriver(
    std::unique_ptr<internal::MusaSymbolLoader> symbol_loader)
    : symbol_loader_(std::move(symbol_loader)) {}

MusaDriver::~MusaDriver() = default;

MusaDriver& MusaDriver::Instance() {
  static MusaDriver* driver = new MusaDriver();
  return *driver;
}

absl::Status MusaDriver::Init() {
  std::call_once(init_once_, [this] { init_status_ = Initialize(); });
  return init_status_;
}

absl::Status MusaDriver::Initialize() {
  if (symbol_loader_ == nullptr) {
    return absl::InvalidArgumentError(
        "MusaDriver requires a non-null symbol loader");
  }
  absl::Status load_status = symbol_loader_->Load();
  if (!load_status.ok()) return load_status;

  BootstrapApi bootstrap;
  absl::StatusOr<MuInitFn> init =
      ResolveDirect<MuInitFn>(*symbol_loader_, "muInit");
  if (!init.ok()) return init.status();
  bootstrap.init = *init;

  absl::StatusOr<void*> get_error_name =
      symbol_loader_->Resolve("muGetErrorName");
  if (get_error_name.ok()) {
    bootstrap.get_error_name =
        reinterpret_cast<MuGetErrorNameFn>(*get_error_name);
  }

  absl::StatusOr<void*> get_error_string =
      symbol_loader_->Resolve("muGetErrorString");
  if (get_error_string.ok()) {
    bootstrap.get_error_string =
        reinterpret_cast<MuGetErrorStringFn>(*get_error_string);
  }

  absl::StatusOr<void*> get_proc_address =
      symbol_loader_->Resolve("muGetProcAddress");
  if (get_proc_address.ok()) {
    bootstrap.get_proc_address =
        reinterpret_cast<MuGetProcAddressFn>(*get_proc_address);
  }

  MUresult init_result = bootstrap.init(0);
  if (init_result != MUSA_SUCCESS) {
    return DriverToStatus(init_result, "muInit(0)", bootstrap.get_error_name,
                          bootstrap.get_error_string);
  }

  auto api = std::make_unique<Api>();
  api->get_error_name = bootstrap.get_error_name;
  api->get_error_string = bootstrap.get_error_string;

#define MUSA_RESOLVE_REQUIRED(field, type, logical_name, ...)     \
  do {                                                            \
    absl::StatusOr<type> resolved = ResolveRequired<type>(        \
        *symbol_loader_, bootstrap, logical_name, {__VA_ARGS__}); \
    if (!resolved.ok()) return resolved.status();                 \
    api->field = *resolved;                                       \
  } while (false)

  MUSA_RESOLVE_REQUIRED(driver_get_version, MuDriverGetVersionFn,
                        "muDriverGetVersion", "muDriverGetVersion");
  MUSA_RESOLVE_REQUIRED(device_get_count, MuDeviceGetCountFn,
                        "muDeviceGetCount", "muDeviceGetCount");
  MUSA_RESOLVE_REQUIRED(device_get, MuDeviceGetFn, "muDeviceGet",
                        "muDeviceGet");
  MUSA_RESOLVE_REQUIRED(device_can_access_peer, MuDeviceCanAccessPeerFn,
                        "muDeviceCanAccessPeer", "muDeviceCanAccessPeer");
  MUSA_RESOLVE_REQUIRED(device_get_p2p_attribute, MuDeviceGetP2PAttributeFn,
                        "muDeviceGetP2PAttribute", "muDeviceGetP2PAttribute");
  MUSA_RESOLVE_REQUIRED(primary_context_retain, MuDevicePrimaryCtxRetainFn,
                        "muDevicePrimaryCtxRetain", "muDevicePrimaryCtxRetain");
  MUSA_RESOLVE_REQUIRED(primary_context_release, MuDevicePrimaryCtxReleaseV2Fn,
                        "muDevicePrimaryCtxRelease",
                        "muDevicePrimaryCtxRelease_v2",
                        "muDevicePrimaryCtxRelease");
  MUSA_RESOLVE_REQUIRED(primary_context_get_state, MuDevicePrimaryCtxGetStateFn,
                        "muDevicePrimaryCtxGetState",
                        "muDevicePrimaryCtxGetState");
  MUSA_RESOLVE_REQUIRED(
      primary_context_set_flags, MuDevicePrimaryCtxSetFlagsV2Fn,
      "muDevicePrimaryCtxSetFlags", "muDevicePrimaryCtxSetFlags_v2",
      "muDevicePrimaryCtxSetFlags");
  MUSA_RESOLVE_REQUIRED(context_set_current, MuCtxSetCurrentFn,
                        "muCtxSetCurrent", "muCtxSetCurrent");
  MUSA_RESOLVE_REQUIRED(context_get_current, MuCtxGetCurrentFn,
                        "muCtxGetCurrent", "muCtxGetCurrent");
  MUSA_RESOLVE_REQUIRED(context_get_device, MuCtxGetDeviceFn, "muCtxGetDevice",
                        "muCtxGetDevice");
  MUSA_RESOLVE_REQUIRED(context_synchronize, MuCtxSynchronizeFn,
                        "muCtxSynchronize", "muCtxSynchronize");
  MUSA_RESOLVE_REQUIRED(context_enable_peer_access, MuCtxEnablePeerAccessFn,
                        "muCtxEnablePeerAccess", "muCtxEnablePeerAccess");
  MUSA_RESOLVE_REQUIRED(pointer_get_attribute, MuPointerGetAttributeFn,
                        "muPointerGetAttribute", "muPointerGetAttribute");
  MUSA_RESOLVE_REQUIRED(memcpy_peer_async, MuMemcpyPeerAsyncFn,
                        "muMemcpyPeerAsync", "muMemcpyPeerAsync");
  MUSA_RESOLVE_REQUIRED(module_load_data, MuModuleLoadDataFn,
                        "muModuleLoadData", "muModuleLoadData");
  MUSA_RESOLVE_REQUIRED(module_unload, MuModuleUnloadFn, "muModuleUnload",
                        "muModuleUnload");
  MUSA_RESOLVE_REQUIRED(module_get_function, MuModuleGetFunctionFn,
                        "muModuleGetFunction", "muModuleGetFunction");
  MUSA_RESOLVE_REQUIRED(module_get_global, MuModuleGetGlobalV2Fn,
                        "muModuleGetGlobal", "muModuleGetGlobal_v2");
  MUSA_RESOLVE_REQUIRED(function_get_attribute, MuFuncGetAttributeFn,
                        "muFuncGetAttribute", "muFuncGetAttribute");
  MUSA_RESOLVE_REQUIRED(occupancy_max_active_blocks,
                        MuOccupancyMaxActiveBlocksPerMultiprocessorFn,
                        "muOccupancyMaxActiveBlocksPerMultiprocessor",
                        "muOccupancyMaxActiveBlocksPerMultiprocessor");
  MUSA_RESOLVE_REQUIRED(launch_kernel, MuLaunchKernelFn, "muLaunchKernel",
                        "muLaunchKernel");
  MUSA_RESOLVE_REQUIRED(memset_d32_async, MuMemsetD32AsyncFn,
                        "muMemsetD32Async", "muMemsetD32Async");

#define MUSA_RESOLVE_OPTIONAL(field, type, logical_name, ...)                  \
  api->field = ResolveOptional<type>(*symbol_loader_, bootstrap, logical_name, \
                                     {__VA_ARGS__})

  MUSA_RESOLVE_OPTIONAL(graph_create, MuGraphCreateFn, "muGraphCreate",
                        "muGraphCreate");
  MUSA_RESOLVE_OPTIONAL(graph_destroy, MuGraphDestroyFn, "muGraphDestroy",
                        "muGraphDestroy");
  MUSA_RESOLVE_OPTIONAL(graph_add_empty_node, MuGraphAddEmptyNodeFn,
                        "muGraphAddEmptyNode", "muGraphAddEmptyNode");
  MUSA_RESOLVE_OPTIONAL(graph_add_kernel_node, MuGraphAddKernelNodeFn,
                        "muGraphAddKernelNode", "muGraphAddKernelNode");
  MUSA_RESOLVE_OPTIONAL(
      graph_kernel_node_set_params, MuGraphKernelNodeSetParamsFn,
      "muGraphKernelNodeSetParams", "muGraphKernelNodeSetParams");
  MUSA_RESOLVE_OPTIONAL(graph_add_memcpy_node, MuGraphAddMemcpyNodeFn,
                        "muGraphAddMemcpyNode", "muGraphAddMemcpyNode");
  MUSA_RESOLVE_OPTIONAL(
      graph_memcpy_node_set_params, MuGraphMemcpyNodeSetParamsFn,
      "muGraphMemcpyNodeSetParams", "muGraphMemcpyNodeSetParams");
  MUSA_RESOLVE_OPTIONAL(graph_add_memset_node, MuGraphAddMemsetNodeFn,
                        "muGraphAddMemsetNode", "muGraphAddMemsetNode");
  MUSA_RESOLVE_OPTIONAL(
      graph_memset_node_set_params, MuGraphMemsetNodeSetParamsFn,
      "muGraphMemsetNodeSetParams", "muGraphMemsetNodeSetParams");
  MUSA_RESOLVE_OPTIONAL(graph_get_nodes, MuGraphGetNodesFn, "muGraphGetNodes",
                        "muGraphGetNodes");
  MUSA_RESOLVE_OPTIONAL(graph_get_root_nodes, MuGraphGetNodesFn,
                        "muGraphGetRootNodes", "muGraphGetRootNodes");
  MUSA_RESOLVE_OPTIONAL(graph_instantiate, MuGraphInstantiateFn,
                        "muGraphInstantiate", "muGraphInstantiate_v2",
                        "muGraphInstantiate");
  MUSA_RESOLVE_OPTIONAL(graph_exec_destroy, MuGraphExecDestroyFn,
                        "muGraphExecDestroy", "muGraphExecDestroy");
  MUSA_RESOLVE_OPTIONAL(graph_launch, MuGraphLaunchFn, "muGraphLaunch",
                        "muGraphLaunch");
  MUSA_RESOLVE_OPTIONAL(stream_synchronize, MuStreamSynchronizeFn,
                        "muStreamSynchronize", "muStreamSynchronize");
  MUSA_RESOLVE_OPTIONAL(stream_begin_capture, MuStreamBeginCaptureFn,
                        "muStreamBeginCapture", "muStreamBeginCapture_v2",
                        "muStreamBeginCapture");
  MUSA_RESOLVE_OPTIONAL(stream_end_capture, MuStreamEndCaptureFn,
                        "muStreamEndCapture", "muStreamEndCapture");

#undef MUSA_RESOLVE_OPTIONAL
#undef MUSA_RESOLVE_REQUIRED

  api_ = std::move(api);
  return absl::OkStatus();
}

absl::Status MusaDriver::ResultStatus(MUresult result,
                                      const char* expression) const {
  return DriverToStatus(result, expression, api_->get_error_name,
                        api_->get_error_string);
}

absl::StatusOr<int> MusaDriver::DriverVersion() {
  absl::Status status = Init();
  if (!status.ok()) return status;
  int version = 0;
  status =
      ResultStatus(api_->driver_get_version(&version), "muDriverGetVersion");
  if (!status.ok()) return status;
  return version;
}

absl::StatusOr<int> MusaDriver::DeviceCount() {
  absl::Status status = Init();
  if (!status.ok()) return status;
  int count = 0;
  status = ResultStatus(api_->device_get_count(&count), "muDeviceGetCount");
  if (!status.ok()) return status;
  return count;
}

absl::StatusOr<MUdevice> MusaDriver::Device(int ordinal) {
  absl::Status status = Init();
  if (!status.ok()) return status;
  MUdevice device = 0;
  status = ResultStatus(api_->device_get(&device, ordinal), "muDeviceGet");
  if (!status.ok()) return status;
  return device;
}

absl::StatusOr<MusaPeerAccessInfo> MusaDriver::PeerAccessInfo(MUdevice source,
                                                              MUdevice peer) {
  if (source == peer) {
    return absl::InvalidArgumentError(
        "MUSA peer access requires two distinct devices");
  }
  absl::Status status = Init();
  if (!status.ok()) return status;

  int can_access_peer = -1;
  status =
      ResultStatus(api_->device_can_access_peer(&can_access_peer, source, peer),
                   "muDeviceCanAccessPeer");
  if (!status.ok()) return status;
  if (can_access_peer != 0 && can_access_peer != 1) {
    return absl::InternalError(absl::StrCat(
        "muDeviceCanAccessPeer returned success with a non-boolean value: ",
        can_access_peer));
  }
  if (can_access_peer == 0) return MusaPeerAccessInfo{};

  auto get_attribute = [&](MUdevice_P2PAttribute attribute,
                           const char* attribute_name) -> absl::StatusOr<int> {
    int value = -1;
    absl::Status attribute_status = ResultStatus(
        api_->device_get_p2p_attribute(&value, attribute, source, peer),
        "muDeviceGetP2PAttribute");
    if (!attribute_status.ok()) {
      return absl::Status(
          attribute_status.code(),
          absl::StrCat(attribute_status.message(), " for ", attribute_name,
                       " from device ", source, " to device ", peer));
    }
    return value;
  };
  auto without_link_attributes = [&](absl::string_view reason) {
    LOG(WARNING) << "MUSA device " << source << " can access peer " << peer
                 << ", but optional link telemetry is unavailable: " << reason;
    return MusaPeerAccessInfo{.can_access_peer = true};
  };

  absl::StatusOr<int> performance_rank = get_attribute(
      MU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK, "performance rank");
  if (!performance_rank.ok()) {
    return without_link_attributes(performance_rank.status().message());
  }
  absl::StatusOr<int> native_atomic = get_attribute(
      MU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED, "native atomic support");
  if (!native_atomic.ok()) {
    return without_link_attributes(native_atomic.status().message());
  }
  absl::StatusOr<int> musa_array =
      get_attribute(MU_DEVICE_P2P_ATTRIBUTE_MUSA_ARRAY_ACCESS_SUPPORTED,
                    "MUSA array access support");
  if (!musa_array.ok()) {
    return without_link_attributes(musa_array.status().message());
  }
  absl::StatusOr<int> mtlink_ports = get_attribute(
      MU_DEVICE_P2P_ATTRIBUTE_MTLINK_PORT_COUNT, "MTLink port count");
  if (!mtlink_ports.ok()) {
    return without_link_attributes(mtlink_ports.status().message());
  }

  if (*performance_rank < 0 || *mtlink_ports < 0) {
    return without_link_attributes(absl::StrCat(
        "muDeviceGetP2PAttribute returned success with a negative topology "
        "value: performance_rank=",
        *performance_rank, ", mtlink_port_count=", *mtlink_ports));
  }
  if ((*native_atomic != 0 && *native_atomic != 1) ||
      (*musa_array != 0 && *musa_array != 1)) {
    return without_link_attributes(absl::StrCat(
        "muDeviceGetP2PAttribute returned success with a non-boolean support "
        "value: native_atomic=",
        *native_atomic, ", musa_array=", *musa_array));
  }
  return MusaPeerAccessInfo{
      .can_access_peer = true,
      .link_attributes_available = true,
      .performance_rank = *performance_rank,
      .native_atomic_supported = *native_atomic != 0,
      .musa_array_access_supported = *musa_array != 0,
      .mtlink_port_count = *mtlink_ports,
  };
}

absl::StatusOr<MUcontext> MusaDriver::RetainPrimaryContext(MUdevice device) {
  absl::Status status = Init();
  if (!status.ok()) return status;
  MUcontext context = nullptr;
  status = ResultStatus(api_->primary_context_retain(&context, device),
                        "muDevicePrimaryCtxRetain");
  if (!status.ok()) return status;
  if (context == nullptr) {
    return absl::InternalError(
        "muDevicePrimaryCtxRetain returned success with a null context");
  }
  return context;
}

absl::Status MusaDriver::ReleasePrimaryContext(MUdevice device) {
  absl::Status status = Init();
  if (!status.ok()) return status;
  return ResultStatus(api_->primary_context_release(device),
                      "muDevicePrimaryCtxRelease");
}

absl::Status MusaDriver::SetPrimaryContextFlags(MUdevice device,
                                                unsigned int flags) {
  absl::Status status = Init();
  if (!status.ok()) return status;
  return ResultStatus(api_->primary_context_set_flags(device, flags),
                      "muDevicePrimaryCtxSetFlags");
}

absl::StatusOr<MusaPrimaryContextState> MusaDriver::PrimaryContextState(
    MUdevice device) {
  absl::Status status = Init();
  if (!status.ok()) return status;
  unsigned int flags = 0;
  int active = 0;
  status =
      ResultStatus(api_->primary_context_get_state(device, &flags, &active),
                   "muDevicePrimaryCtxGetState");
  if (!status.ok()) return status;
  return MusaPrimaryContextState{flags, active != 0};
}

absl::Status MusaDriver::SetCurrentContext(MUcontext context) {
  absl::Status status = Init();
  if (!status.ok()) return status;
  return ResultStatus(api_->context_set_current(context), "muCtxSetCurrent");
}

absl::StatusOr<MUcontext> MusaDriver::CurrentContext() {
  absl::Status status = Init();
  if (!status.ok()) return status;
  MUcontext context = nullptr;
  status = ResultStatus(api_->context_get_current(&context), "muCtxGetCurrent");
  if (!status.ok()) return status;
  return context;
}

absl::StatusOr<MUdevice> MusaDriver::CurrentDevice() {
  absl::Status status = Init();
  if (!status.ok()) return status;
  MUdevice device = 0;
  status = ResultStatus(api_->context_get_device(&device), "muCtxGetDevice");
  if (!status.ok()) return status;
  return device;
}

absl::Status MusaDriver::SynchronizeContext() {
  absl::Status status = Init();
  if (!status.ok()) return status;
  return ResultStatus(api_->context_synchronize(), "muCtxSynchronize");
}

absl::Status MusaDriver::EnablePeerAccess(MUcontext peer_context) {
  if (peer_context == nullptr) {
    return absl::InvalidArgumentError(
        "muCtxEnablePeerAccess requires a non-null peer context");
  }
  absl::Status status = Init();
  if (!status.ok()) return status;
  MUresult result =
      api_->context_enable_peer_access(peer_context, MU_PEERACCESS_DEFAULT);
  if (result == MUSA_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
    return absl::OkStatus();
  }
  return ResultStatus(result, "muCtxEnablePeerAccess");
}

absl::StatusOr<MUcontext> MusaDriver::ContextForPointer(MUdeviceptr pointer) {
  if (pointer == 0) {
    return absl::InvalidArgumentError(
        "muPointerGetAttribute requires a non-null pointer");
  }
  absl::Status status = Init();
  if (!status.ok()) return status;
  MUcontext context = nullptr;
  status = ResultStatus(api_->pointer_get_attribute(
                            &context, MU_POINTER_ATTRIBUTE_CONTEXT, pointer),
                        "muPointerGetAttribute");
  if (!status.ok()) return status;
  if (context == nullptr) {
    return absl::InternalError(
        "muPointerGetAttribute returned success with a null context");
  }
  return context;
}

absl::Status MusaDriver::MemcpyPeerAsync(MUdeviceptr destination,
                                         MUcontext destination_context,
                                         MUdeviceptr source,
                                         MUcontext source_context,
                                         uint64_t bytes, MUstream stream) {
  if (destination == 0 || source == 0) {
    return absl::InvalidArgumentError(
        "muMemcpyPeerAsync requires non-null source and destination pointers");
  }
  if (destination_context == nullptr || source_context == nullptr) {
    return absl::InvalidArgumentError(
        "muMemcpyPeerAsync requires non-null source and destination contexts");
  }
  absl::Status status = Init();
  if (!status.ok()) return status;
  return ResultStatus(
      api_->memcpy_peer_async(destination, destination_context, source,
                              source_context, bytes, stream),
      "muMemcpyPeerAsync");
}

absl::StatusOr<MUmodule> MusaDriver::LoadModuleData(const void* image) {
  if (image == nullptr) {
    return absl::InvalidArgumentError(
        "muModuleLoadData requires a non-null image");
  }
  absl::Status status = Init();
  if (!status.ok()) return status;
  MUmodule module = nullptr;
  status =
      ResultStatus(api_->module_load_data(&module, image), "muModuleLoadData");
  if (!status.ok()) return status;
  if (module == nullptr) {
    return absl::InternalError(
        "muModuleLoadData returned success with a null module");
  }
  return module;
}

absl::Status MusaDriver::UnloadModule(MUmodule module) {
  if (module == nullptr) {
    return absl::InvalidArgumentError(
        "muModuleUnload requires a non-null module");
  }
  absl::Status status = Init();
  if (!status.ok()) return status;
  return ResultStatus(api_->module_unload(module), "muModuleUnload");
}

absl::StatusOr<MUfunction> MusaDriver::GetModuleFunction(MUmodule module,
                                                         const char* name) {
  if (module == nullptr) {
    return absl::InvalidArgumentError(
        "muModuleGetFunction requires a non-null module");
  }
  if (name == nullptr || name[0] == '\0') {
    return absl::InvalidArgumentError(
        "muModuleGetFunction requires a non-empty name");
  }
  absl::Status status = Init();
  if (!status.ok()) return status;
  MUfunction function = nullptr;
  status = ResultStatus(api_->module_get_function(&function, module, name),
                        "muModuleGetFunction");
  if (!status.ok()) return status;
  if (function == nullptr) {
    return absl::InternalError(
        "muModuleGetFunction returned success with a null function");
  }
  return function;
}

absl::StatusOr<MusaModuleGlobal> MusaDriver::GetModuleGlobal(MUmodule module,
                                                             const char* name) {
  if (module == nullptr) {
    return absl::InvalidArgumentError(
        "muModuleGetGlobal requires a non-null module");
  }
  if (name == nullptr || name[0] == '\0') {
    return absl::InvalidArgumentError(
        "muModuleGetGlobal requires a non-empty name");
  }
  absl::Status status = Init();
  if (!status.ok()) return status;
  MUdeviceptr address = 0;
  size_t size = 0;
  status = ResultStatus(api_->module_get_global(&address, &size, module, name),
                        "muModuleGetGlobal");
  if (!status.ok()) return status;
  if (address == 0) {
    return absl::InternalError(
        "muModuleGetGlobal returned success with a null address");
  }
  if (size == 0) {
    return absl::InternalError(
        "muModuleGetGlobal returned success with a zero-byte symbol");
  }
  return MusaModuleGlobal{address, size};
}

absl::StatusOr<int> MusaDriver::FunctionAttribute(
    MUfunction function, MUfunction_attribute attribute) {
  if (function == nullptr) {
    return absl::InvalidArgumentError(
        "muFuncGetAttribute requires a non-null function");
  }
  absl::Status status = Init();
  if (!status.ok()) return status;
  int value = 0;
  status =
      ResultStatus(api_->function_get_attribute(&value, attribute, function),
                   "muFuncGetAttribute");
  if (!status.ok()) return status;
  return value;
}

absl::StatusOr<int> MusaDriver::MaxActiveBlocksPerMultiprocessor(
    MUfunction function, int block_size, size_t dynamic_shared_memory_bytes) {
  if (function == nullptr) {
    return absl::InvalidArgumentError(
        "muOccupancyMaxActiveBlocksPerMultiprocessor requires a non-null "
        "function");
  }
  if (block_size <= 0) {
    return absl::InvalidArgumentError(
        "muOccupancyMaxActiveBlocksPerMultiprocessor requires a positive "
        "block size");
  }
  absl::Status status = Init();
  if (!status.ok()) return status;
  int blocks = 0;
  status = ResultStatus(
      api_->occupancy_max_active_blocks(&blocks, function, block_size,
                                        dynamic_shared_memory_bytes),
      "muOccupancyMaxActiveBlocksPerMultiprocessor");
  if (!status.ok()) return status;
  if (blocks < 0) {
    return absl::InternalError(
        "muOccupancyMaxActiveBlocksPerMultiprocessor returned success with a "
        "negative block count");
  }
  return blocks;
}

absl::Status MusaDriver::LaunchKernel(
    MUfunction function, unsigned int grid_dim_x, unsigned int grid_dim_y,
    unsigned int grid_dim_z, unsigned int block_dim_x, unsigned int block_dim_y,
    unsigned int block_dim_z, unsigned int shared_memory_bytes, MUstream stream,
    void** kernel_parameters, void** extra) {
  if (function == nullptr) {
    return absl::InvalidArgumentError(
        "muLaunchKernel requires a non-null function");
  }
  if (grid_dim_x == 0 || grid_dim_y == 0 || grid_dim_z == 0 ||
      block_dim_x == 0 || block_dim_y == 0 || block_dim_z == 0) {
    return absl::InvalidArgumentError(
        "muLaunchKernel requires non-zero grid and block dimensions");
  }
  absl::Status status = Init();
  if (!status.ok()) return status;
  return ResultStatus(api_->launch_kernel(function, grid_dim_x, grid_dim_y,
                                          grid_dim_z, block_dim_x, block_dim_y,
                                          block_dim_z, shared_memory_bytes,
                                          stream, kernel_parameters, extra),
                      "muLaunchKernel");
}

absl::Status MusaDriver::MemsetD32Async(MUdeviceptr destination, uint32_t value,
                                        size_t count, MUstream stream) {
  if (destination == 0) {
    return absl::InvalidArgumentError(
        "muMemsetD32Async requires a non-null destination");
  }
  absl::Status status = Init();
  if (!status.ok()) return status;
  return ResultStatus(api_->memset_d32_async(destination, value, count, stream),
                      "muMemsetD32Async");
}

bool MusaDriver::GraphsAvailable() {
  if (!Init().ok()) return false;
  return api_->graph_create != nullptr && api_->graph_destroy != nullptr &&
         api_->graph_add_empty_node != nullptr &&
         api_->graph_add_kernel_node != nullptr &&
         api_->graph_kernel_node_set_params != nullptr &&
         api_->graph_add_memcpy_node != nullptr &&
         api_->graph_memcpy_node_set_params != nullptr &&
         api_->graph_add_memset_node != nullptr &&
         api_->graph_memset_node_set_params != nullptr &&
         api_->graph_get_nodes != nullptr &&
         api_->graph_get_root_nodes != nullptr &&
         api_->graph_instantiate != nullptr &&
         api_->graph_exec_destroy != nullptr && api_->graph_launch != nullptr &&
         api_->stream_synchronize != nullptr &&
         api_->stream_begin_capture != nullptr &&
         api_->stream_end_capture != nullptr;
}

absl::StatusOr<MUgraph> MusaDriver::GraphCreate() {
  absl::Status status = Init();
  if (!status.ok()) return status;
  if (api_->graph_create == nullptr) {
    return absl::UnimplementedError(
        "MUSA graphs require muGraphCreate in libmusa");
  }
  MUgraph graph = nullptr;
  status =
      ResultStatus(api_->graph_create(&graph, /*flags=*/0), "muGraphCreate");
  if (!status.ok()) return status;
  if (graph == nullptr) {
    return absl::InternalError(
        "muGraphCreate returned success with a null graph");
  }
  return graph;
}

absl::Status MusaDriver::GraphDestroy(MUgraph graph) {
  if (graph == nullptr) return absl::OkStatus();
  absl::Status status = Init();
  if (!status.ok()) return status;
  if (api_->graph_destroy == nullptr) {
    return absl::UnimplementedError(
        "MUSA graphs require muGraphDestroy in libmusa");
  }
  return ResultStatus(api_->graph_destroy(graph), "muGraphDestroy");
}

absl::StatusOr<MUgraphNode> MusaDriver::GraphAddEmptyNode(
    MUgraph graph, absl::Span<const MUgraphNode> dependencies) {
  absl::Status status = Init();
  if (!status.ok()) return status;
  if (api_->graph_add_empty_node == nullptr) {
    return absl::UnimplementedError(
        "MUSA graphs require muGraphAddEmptyNode in libmusa");
  }
  MUgraphNode node = nullptr;
  status =
      ResultStatus(api_->graph_add_empty_node(&node, graph, dependencies.data(),
                                              dependencies.size()),
                   "muGraphAddEmptyNode");
  if (!status.ok()) return status;
  if (node == nullptr) {
    return absl::InternalError(
        "muGraphAddEmptyNode returned success with a null node");
  }
  return node;
}

absl::StatusOr<MUgraphNode> MusaDriver::GraphAddKernelNode(
    MUgraph graph, absl::Span<const MUgraphNode> dependencies,
    const MUSA_KERNEL_NODE_PARAMS& params) {
  absl::Status status = Init();
  if (!status.ok()) return status;
  if (api_->graph_add_kernel_node == nullptr) {
    return absl::UnimplementedError(
        "MUSA graphs require muGraphAddKernelNode in libmusa");
  }
  MUgraphNode node = nullptr;
  status = ResultStatus(
      api_->graph_add_kernel_node(&node, graph, dependencies.data(),
                                  dependencies.size(), &params),
      "muGraphAddKernelNode");
  if (!status.ok()) return status;
  if (node == nullptr) {
    return absl::InternalError(
        "muGraphAddKernelNode returned success with a null node");
  }
  return node;
}

absl::Status MusaDriver::GraphKernelNodeSetParams(
    MUgraphNode node, const MUSA_KERNEL_NODE_PARAMS& params) {
  absl::Status status = Init();
  if (!status.ok()) return status;
  if (api_->graph_kernel_node_set_params == nullptr) {
    return absl::UnimplementedError(
        "MUSA graphs require muGraphKernelNodeSetParams in libmusa");
  }
  return ResultStatus(api_->graph_kernel_node_set_params(node, &params),
                      "muGraphKernelNodeSetParams");
}

absl::StatusOr<MUgraphNode> MusaDriver::GraphAddMemcpyD2DNode(
    MUgraph graph, absl::Span<const MUgraphNode> dependencies,
    MUdeviceptr destination, MUdeviceptr source, size_t bytes) {
  absl::Status status = Init();
  if (!status.ok()) return status;
  if (api_->graph_add_memcpy_node == nullptr) {
    return absl::UnimplementedError(
        "MUSA graphs require muGraphAddMemcpyNode in libmusa");
  }
  MUcontext context = nullptr;
  status = ResultStatus(api_->context_get_current(&context), "muCtxGetCurrent");
  if (!status.ok()) return status;
  if (context == nullptr) {
    return absl::FailedPreconditionError(
        "muGraphAddMemcpyNode requires a current MUSA context");
  }
  MUSA_MEMCPY3D params = {};
  params.srcMemoryType = MU_MEMORYTYPE_DEVICE;
  params.srcDevice = source;
  params.srcPitch = bytes;
  params.srcHeight = 1;
  params.dstMemoryType = MU_MEMORYTYPE_DEVICE;
  params.dstDevice = destination;
  params.dstPitch = bytes;
  params.dstHeight = 1;
  params.WidthInBytes = bytes;
  params.Height = 1;
  params.Depth = 1;
  MUgraphNode node = nullptr;
  status = ResultStatus(
      api_->graph_add_memcpy_node(&node, graph, dependencies.data(),
                                  dependencies.size(), &params, context),
      "muGraphAddMemcpyNode");
  if (!status.ok()) return status;
  if (node == nullptr) {
    return absl::InternalError(
        "muGraphAddMemcpyNode returned success with a null node");
  }
  return node;
}

absl::Status MusaDriver::GraphMemcpyD2DNodeSetParams(MUgraphNode node,
                                                     MUdeviceptr destination,
                                                     MUdeviceptr source,
                                                     size_t bytes) {
  absl::Status status = Init();
  if (!status.ok()) return status;
  if (api_->graph_memcpy_node_set_params == nullptr) {
    return absl::UnimplementedError(
        "MUSA graphs require muGraphMemcpyNodeSetParams in libmusa");
  }
  MUSA_MEMCPY3D params = {};
  params.srcMemoryType = MU_MEMORYTYPE_DEVICE;
  params.srcDevice = source;
  params.srcPitch = bytes;
  params.srcHeight = 1;
  params.dstMemoryType = MU_MEMORYTYPE_DEVICE;
  params.dstDevice = destination;
  params.dstPitch = bytes;
  params.dstHeight = 1;
  params.WidthInBytes = bytes;
  params.Height = 1;
  params.Depth = 1;
  return ResultStatus(api_->graph_memcpy_node_set_params(node, &params),
                      "muGraphMemcpyNodeSetParams");
}

absl::StatusOr<MUgraphNode> MusaDriver::GraphAddMemsetNode(
    MUgraph graph, absl::Span<const MUgraphNode> dependencies,
    const MUSA_MEMSET_NODE_PARAMS& params) {
  absl::Status status = Init();
  if (!status.ok()) return status;
  if (api_->graph_add_memset_node == nullptr) {
    return absl::UnimplementedError(
        "MUSA graphs require muGraphAddMemsetNode in libmusa");
  }
  MUcontext context = nullptr;
  status = ResultStatus(api_->context_get_current(&context), "muCtxGetCurrent");
  if (!status.ok()) return status;
  if (context == nullptr) {
    return absl::FailedPreconditionError(
        "muGraphAddMemsetNode requires a current MUSA context");
  }
  MUgraphNode node = nullptr;
  status = ResultStatus(
      api_->graph_add_memset_node(&node, graph, dependencies.data(),
                                  dependencies.size(), &params, context),
      "muGraphAddMemsetNode");
  if (!status.ok()) return status;
  if (node == nullptr) {
    return absl::InternalError(
        "muGraphAddMemsetNode returned success with a null node");
  }
  return node;
}

absl::Status MusaDriver::GraphMemsetNodeSetParams(
    MUgraphNode node, const MUSA_MEMSET_NODE_PARAMS& params) {
  absl::Status status = Init();
  if (!status.ok()) return status;
  if (api_->graph_memset_node_set_params == nullptr) {
    return absl::UnimplementedError(
        "MUSA graphs require muGraphMemsetNodeSetParams in libmusa");
  }
  return ResultStatus(api_->graph_memset_node_set_params(node, &params),
                      "muGraphMemsetNodeSetParams");
}

absl::StatusOr<size_t> MusaDriver::GraphNodeCount(MUgraph graph) {
  absl::Status status = Init();
  if (!status.ok()) return status;
  if (api_->graph_get_nodes == nullptr) {
    return absl::UnimplementedError(
        "MUSA graphs require muGraphGetNodes in libmusa");
  }
  size_t count = 0;
  status = ResultStatus(api_->graph_get_nodes(graph, nullptr, &count),
                        "muGraphGetNodes");
  if (!status.ok()) return status;
  return count;
}

absl::StatusOr<size_t> MusaDriver::GraphRootNodeCount(MUgraph graph) {
  absl::Status status = Init();
  if (!status.ok()) return status;
  if (api_->graph_get_root_nodes == nullptr) {
    return absl::UnimplementedError(
        "MUSA graphs require muGraphGetRootNodes in libmusa");
  }
  size_t count = 0;
  status = ResultStatus(api_->graph_get_root_nodes(graph, nullptr, &count),
                        "muGraphGetRootNodes");
  if (!status.ok()) return status;
  return count;
}

absl::StatusOr<MUgraphExec> MusaDriver::GraphInstantiate(MUgraph graph) {
  absl::Status status = Init();
  if (!status.ok()) return status;
  if (api_->graph_instantiate == nullptr) {
    return absl::UnimplementedError(
        "MUSA graphs require muGraphInstantiate in libmusa");
  }
  MUgraphExec executable = nullptr;
  MUgraphNode error_node = nullptr;
  std::array<char, 1024> log = {};
  status = ResultStatus(api_->graph_instantiate(&executable, graph, &error_node,
                                                log.data(), log.size()),
                        "muGraphInstantiate");
  if (!status.ok()) {
    return absl::Status(status.code(),
                        absl::StrCat(status.message(), "; error_node=",
                                     reinterpret_cast<uintptr_t>(error_node),
                                     "; log=", log.data()));
  }
  if (executable == nullptr) {
    return absl::InternalError(
        "muGraphInstantiate returned success with a null executable");
  }
  return executable;
}

absl::Status MusaDriver::GraphExecDestroy(MUgraphExec executable) {
  if (executable == nullptr) return absl::OkStatus();
  absl::Status status = Init();
  if (!status.ok()) return status;
  if (api_->graph_exec_destroy == nullptr) {
    return absl::UnimplementedError(
        "MUSA graphs require muGraphExecDestroy in libmusa");
  }
  return ResultStatus(api_->graph_exec_destroy(executable),
                      "muGraphExecDestroy");
}

absl::Status MusaDriver::GraphLaunch(MUgraphExec executable, MUstream stream) {
  absl::Status status = Init();
  if (!status.ok()) return status;
  if (api_->graph_launch == nullptr) {
    return absl::UnimplementedError(
        "MUSA graphs require muGraphLaunch in libmusa");
  }
  return ResultStatus(api_->graph_launch(executable, stream), "muGraphLaunch");
}

absl::Status MusaDriver::StreamSynchronize(MUstream stream) {
  absl::Status status = Init();
  if (!status.ok()) return status;
  if (api_->stream_synchronize == nullptr) {
    return absl::UnimplementedError(
        "MUSA graph updates require muStreamSynchronize in libmusa");
  }
  return ResultStatus(api_->stream_synchronize(stream), "muStreamSynchronize");
}

absl::Status MusaDriver::StreamBeginCapture(MUstream stream) {
  absl::Status status = Init();
  if (!status.ok()) return status;
  if (api_->stream_begin_capture == nullptr) {
    return absl::UnimplementedError(
        "MUSA graph capture requires muStreamBeginCapture in libmusa");
  }
  return ResultStatus(
      api_->stream_begin_capture(stream, MU_STREAM_CAPTURE_MODE_THREAD_LOCAL),
      "muStreamBeginCapture");
}

absl::StatusOr<MUgraph> MusaDriver::StreamEndCapture(MUstream stream) {
  absl::Status status = Init();
  if (!status.ok()) return status;
  if (api_->stream_end_capture == nullptr) {
    return absl::UnimplementedError(
        "MUSA graph capture requires muStreamEndCapture in libmusa");
  }
  MUgraph graph = nullptr;
  status = ResultStatus(api_->stream_end_capture(stream, &graph),
                        "muStreamEndCapture");
  if (!status.ok()) return status;
  if (graph == nullptr) {
    return absl::InternalError(
        "muStreamEndCapture returned success with a null graph");
  }
  return graph;
}

}  // namespace stream_executor::musa
