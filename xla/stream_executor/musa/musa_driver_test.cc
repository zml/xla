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

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "musa.h"
#include "xla/stream_executor/musa/musa_dso_loader.h"
#include "xla/stream_executor/musa/musa_status.h"

namespace stream_executor::musa {
namespace {

class FakeSymbolLoader final : public internal::MusaSymbolLoader {
 public:
  absl::Status Load() override {
    ++load_calls;
    return load_status;
  }

  absl::StatusOr<void*> Resolve(absl::string_view symbol) const override {
    std::lock_guard<std::mutex> lock(mu_);
    ++resolve_calls_[std::string(symbol)];
    auto it = symbols_.find(std::string(symbol));
    if (it == symbols_.end()) {
      return absl::NotFoundError(std::string(symbol) + " absent from fake DSO");
    }
    return it->second;
  }

  absl::string_view loaded_path() const override { return path; }

  template <typename Fn>
  void Add(absl::string_view name, Fn function) {
    std::lock_guard<std::mutex> lock(mu_);
    symbols_[std::string(name)] = reinterpret_cast<void*>(function);
  }

  void Remove(absl::string_view name) {
    std::lock_guard<std::mutex> lock(mu_);
    symbols_.erase(std::string(name));
  }

  int ResolveCalls(absl::string_view name) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = resolve_calls_.find(std::string(name));
    return it == resolve_calls_.end() ? 0 : it->second;
  }

  std::atomic<int> load_calls{0};
  absl::Status load_status = absl::OkStatus();
  std::string path = "/fake/libmusa.so.1.5";

 private:
  mutable std::mutex mu_;
  std::unordered_map<std::string, void*> symbols_;
  mutable std::unordered_map<std::string, int> resolve_calls_;
};

struct FakeApiState {
  MUresult init_result = MUSA_SUCCESS;
  std::atomic<int> init_calls{0};
  int driver_version = 10504;
  int device_count = 2;
  MUcontext retained_context = reinterpret_cast<MUcontext>(0x1234);
  unsigned int primary_flags = 7;
  int primary_active = 1;
  MUcontext current_context = reinterpret_cast<MUcontext>(0x5678);
  MUdevice current_device = 11;
  int peer_can_access = 1;
  MUresult peer_can_access_result = MUSA_SUCCESS;
  MUdevice last_peer_source = -1;
  MUdevice last_peer_target = -1;
  std::atomic<int> peer_can_access_calls{0};
  std::unordered_map<int, int> p2p_attributes = {
      {MU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK, 7},
      {MU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED, 1},
      {MU_DEVICE_P2P_ATTRIBUTE_MUSA_ARRAY_ACCESS_SUPPORTED, 0},
      {MU_DEVICE_P2P_ATTRIBUTE_MTLINK_PORT_COUNT, 2},
  };
  std::vector<MUdevice_P2PAttribute> queried_p2p_attributes;
  MUresult p2p_attribute_result = MUSA_SUCCESS;
  MUcontext last_enabled_peer_context = nullptr;
  unsigned int last_peer_flags = ~0u;
  std::atomic<int> enable_peer_calls{0};
  MUresult enable_peer_result = MUSA_SUCCESS;
  MUcontext pointer_context = reinterpret_cast<MUcontext>(0x3344);
  MUdeviceptr last_pointer = 0;
  MUpointer_attribute last_pointer_attribute = MU_POINTER_ATTRIBUTE_CONTEXT;
  std::atomic<int> pointer_attribute_calls{0};
  MUresult pointer_attribute_result = MUSA_SUCCESS;
  MUdeviceptr last_copy_destination = 0;
  MUcontext last_copy_destination_context = nullptr;
  MUdeviceptr last_copy_source = 0;
  MUcontext last_copy_source_context = nullptr;
  size_t last_copy_bytes = 0;
  MUstream last_copy_stream = nullptr;
  std::atomic<int> peer_copy_calls{0};
  MUresult peer_copy_result = MUSA_SUCCESS;
  std::atomic<int> release_calls{0};
  MUresult release_result = MUSA_SUCCESS;
  MUresult set_flags_result = MUSA_SUCCESS;
  MUmodule module = reinterpret_cast<MUmodule>(0x9876);
  const void* last_module_image = nullptr;
  MUmodule last_unloaded_module = nullptr;
  std::atomic<int> module_load_calls{0};
  std::atomic<int> module_unload_calls{0};
  MUresult module_load_result = MUSA_SUCCESS;
  MUresult module_unload_result = MUSA_SUCCESS;
  MUfunction function = reinterpret_cast<MUfunction>(0x2468);
  MUmodule last_lookup_module = nullptr;
  std::string last_lookup_name;
  MUresult module_get_function_result = MUSA_SUCCESS;
  MUdeviceptr global_address = static_cast<MUdeviceptr>(0xabc000);
  size_t global_size = 64;
  MUresult module_get_global_result = MUSA_SUCCESS;
  int function_attribute = 32;
  MUfunction_attribute last_function_attribute =
      MU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK;
  MUresult function_attribute_result = MUSA_SUCCESS;
  int occupancy_blocks = 3;
  int last_occupancy_block_size = 0;
  size_t last_occupancy_dynamic_shared_memory_bytes = 0;
  MUresult occupancy_result = MUSA_SUCCESS;

  struct LaunchArguments {
    MUfunction function = nullptr;
    unsigned int grid_dim_x = 0;
    unsigned int grid_dim_y = 0;
    unsigned int grid_dim_z = 0;
    unsigned int block_dim_x = 0;
    unsigned int block_dim_y = 0;
    unsigned int block_dim_z = 0;
    unsigned int shared_memory_bytes = 0;
    MUstream stream = nullptr;
    void** kernel_parameters = nullptr;
    void** extra = nullptr;
  } last_launch;
  std::atomic<int> launch_calls{0};
  MUresult launch_result = MUSA_SUCCESS;

  MUdeviceptr last_memset_destination = 0;
  uint32_t last_memset_value = 0;
  size_t last_memset_count = 0;
  MUstream last_memset_stream = nullptr;
  std::atomic<int> memset_calls{0};
  MUresult memset_result = MUSA_SUCCESS;

  std::unordered_map<std::string, void*> gpa_symbols;
  bool gpa_success_null = false;
  std::atomic<int> gpa_calls{0};
  std::atomic<int> last_gpa_version{0};
  std::atomic<uint64_t> last_gpa_flags{0};

  bool block_device_count = false;
  std::mutex device_count_mu;
  std::condition_variable device_count_cv;
  int active_device_count_calls = 0;
  int device_count_arrivals = 0;
  int max_active_device_count_calls = 0;
};

FakeApiState* g_state = nullptr;

MUresult MUSAAPI FakeMuInit(unsigned int flags) {
  EXPECT_EQ(flags, 0);
  ++g_state->init_calls;
  return g_state->init_result;
}

MUresult MUSAAPI FakeMuGetErrorName(MUresult error, const char** name) {
  if (name == nullptr) return MUSA_ERROR_INVALID_VALUE;
  switch (error) {
    case MUSA_ERROR_OUT_OF_MEMORY:
      *name = "MUSA_ERROR_OUT_OF_MEMORY";
      break;
    case MUSA_ERROR_INVALID_VALUE:
      *name = "MUSA_ERROR_INVALID_VALUE";
      break;
    case MUSA_ERROR_NOT_INITIALIZED:
      *name = "MUSA_ERROR_NOT_INITIALIZED";
      break;
    default:
      *name = "MUSA_ERROR_TEST";
      break;
  }
  return MUSA_SUCCESS;
}

MUresult MUSAAPI FakeMuGetErrorString(MUresult error,
                                      const char** description) {
  if (description == nullptr) return MUSA_ERROR_INVALID_VALUE;
  switch (error) {
    case MUSA_ERROR_OUT_OF_MEMORY:
      *description = "out of memory";
      break;
    case MUSA_ERROR_INVALID_VALUE:
      *description = "invalid value";
      break;
    case MUSA_ERROR_NOT_INITIALIZED:
      *description = "driver is not initialized";
      break;
    default:
      *description = "test error";
      break;
  }
  return MUSA_SUCCESS;
}

MUresult MUSAAPI FakeMuGetProcAddress(const char* symbol, void** address,
                                      int version, muuint64_t flags) {
  ++g_state->gpa_calls;
  g_state->last_gpa_version = version;
  g_state->last_gpa_flags = flags;
  if (g_state->gpa_success_null) {
    *address = nullptr;
    return MUSA_SUCCESS;
  }
  auto it = g_state->gpa_symbols.find(symbol);
  *address = it == g_state->gpa_symbols.end() ? nullptr : it->second;
  return it == g_state->gpa_symbols.end() ? MUSA_ERROR_NOT_FOUND : MUSA_SUCCESS;
}

MUresult MUSAAPI FakeMuDriverGetVersion(int* version) {
  *version = g_state->driver_version;
  return MUSA_SUCCESS;
}

MUresult MUSAAPI FakeMuDeviceGetCount(int* count) {
  if (g_state->block_device_count) {
    std::unique_lock<std::mutex> lock(g_state->device_count_mu);
    ++g_state->active_device_count_calls;
    ++g_state->device_count_arrivals;
    g_state->max_active_device_count_calls =
        std::max(g_state->max_active_device_count_calls,
                 g_state->active_device_count_calls);
    g_state->device_count_cv.notify_all();
    g_state->device_count_cv.wait(
        lock, [] { return g_state->device_count_arrivals >= 2; });
    --g_state->active_device_count_calls;
    g_state->device_count_cv.notify_all();
  }
  *count = g_state->device_count;
  return MUSA_SUCCESS;
}

MUresult MUSAAPI FakeMuDeviceGet(MUdevice* device, int ordinal) {
  *device = ordinal + 10;
  return MUSA_SUCCESS;
}

MUresult MUSAAPI FakeMuDeviceCanAccessPeer(int* can_access_peer,
                                           MUdevice source, MUdevice peer) {
  ++g_state->peer_can_access_calls;
  g_state->last_peer_source = source;
  g_state->last_peer_target = peer;
  *can_access_peer = g_state->peer_can_access;
  return g_state->peer_can_access_result;
}

MUresult MUSAAPI FakeMuDeviceGetP2PAttribute(
    int* value, MUdevice_P2PAttribute attribute, MUdevice source,
    MUdevice peer) {
  g_state->last_peer_source = source;
  g_state->last_peer_target = peer;
  g_state->queried_p2p_attributes.push_back(attribute);
  auto found = g_state->p2p_attributes.find(attribute);
  *value = found == g_state->p2p_attributes.end() ? -1 : found->second;
  return g_state->p2p_attribute_result;
}

MUresult MUSAAPI FakeMuDevicePrimaryCtxRetain(MUcontext* context,
                                              MUdevice device) {
  EXPECT_EQ(device, 10);
  *context = g_state->retained_context;
  return MUSA_SUCCESS;
}

MUresult MUSAAPI FakeMuDevicePrimaryCtxRelease(MUdevice device) {
  EXPECT_EQ(device, 10);
  ++g_state->release_calls;
  return g_state->release_result;
}

MUresult MUSAAPI FakeMuDevicePrimaryCtxGetState(MUdevice device,
                                                unsigned int* flags,
                                                int* active) {
  EXPECT_EQ(device, 10);
  *flags = g_state->primary_flags;
  *active = g_state->primary_active;
  return MUSA_SUCCESS;
}

MUresult MUSAAPI FakeMuDevicePrimaryCtxSetFlags(MUdevice device,
                                                unsigned int flags) {
  EXPECT_EQ(device, 10);
  g_state->primary_flags = flags;
  return g_state->set_flags_result;
}

MUresult MUSAAPI FakeMuCtxSetCurrent(MUcontext context) {
  g_state->current_context = context;
  return MUSA_SUCCESS;
}

MUresult MUSAAPI FakeMuCtxGetCurrent(MUcontext* context) {
  *context = g_state->current_context;
  return MUSA_SUCCESS;
}

MUresult MUSAAPI FakeMuCtxGetDevice(MUdevice* device) {
  *device = g_state->current_device;
  return MUSA_SUCCESS;
}

MUresult MUSAAPI FakeMuCtxSynchronize() { return MUSA_SUCCESS; }

MUresult MUSAAPI FakeMuCtxEnablePeerAccess(MUcontext peer_context,
                                            unsigned int flags) {
  ++g_state->enable_peer_calls;
  g_state->last_enabled_peer_context = peer_context;
  g_state->last_peer_flags = flags;
  return g_state->enable_peer_result;
}

MUresult MUSAAPI FakeMuPointerGetAttribute(void* value,
                                           MUpointer_attribute attribute,
                                           MUdeviceptr pointer) {
  ++g_state->pointer_attribute_calls;
  g_state->last_pointer = pointer;
  g_state->last_pointer_attribute = attribute;
  if (g_state->pointer_attribute_result == MUSA_SUCCESS) {
    *static_cast<MUcontext*>(value) = g_state->pointer_context;
  }
  return g_state->pointer_attribute_result;
}

MUresult MUSAAPI FakeMuMemcpyPeerAsync(
    MUdeviceptr destination, MUcontext destination_context, MUdeviceptr source,
    MUcontext source_context, size_t bytes, MUstream stream) {
  ++g_state->peer_copy_calls;
  g_state->last_copy_destination = destination;
  g_state->last_copy_destination_context = destination_context;
  g_state->last_copy_source = source;
  g_state->last_copy_source_context = source_context;
  g_state->last_copy_bytes = bytes;
  g_state->last_copy_stream = stream;
  return g_state->peer_copy_result;
}

MUresult MUSAAPI FakeMuModuleLoadData(MUmodule* module, const void* image) {
  ++g_state->module_load_calls;
  g_state->last_module_image = image;
  *module = g_state->module;
  return g_state->module_load_result;
}

MUresult MUSAAPI FakeMuModuleUnload(MUmodule module) {
  ++g_state->module_unload_calls;
  g_state->last_unloaded_module = module;
  return g_state->module_unload_result;
}

MUresult MUSAAPI FakeMuModuleGetFunction(MUfunction* function, MUmodule module,
                                         const char* name) {
  g_state->last_lookup_module = module;
  g_state->last_lookup_name = name;
  *function = g_state->function;
  return g_state->module_get_function_result;
}

MUresult MUSAAPI FakeMuModuleGetGlobal(MUdeviceptr* address, size_t* size,
                                       MUmodule module, const char* name) {
  g_state->last_lookup_module = module;
  g_state->last_lookup_name = name;
  *address = g_state->global_address;
  *size = g_state->global_size;
  return g_state->module_get_global_result;
}

MUresult MUSAAPI FakeMuFuncGetAttribute(int* value,
                                        MUfunction_attribute attribute,
                                        MUfunction function) {
  EXPECT_EQ(function, g_state->function);
  g_state->last_function_attribute = attribute;
  *value = g_state->function_attribute;
  return g_state->function_attribute_result;
}

MUresult MUSAAPI FakeMuOccupancyMaxActiveBlocksPerMultiprocessor(
    int* blocks, MUfunction function, int block_size,
    size_t dynamic_shared_memory_bytes) {
  EXPECT_EQ(function, g_state->function);
  g_state->last_occupancy_block_size = block_size;
  g_state->last_occupancy_dynamic_shared_memory_bytes =
      dynamic_shared_memory_bytes;
  *blocks = g_state->occupancy_blocks;
  return g_state->occupancy_result;
}

MUresult MUSAAPI FakeMuLaunchKernel(
    MUfunction function, unsigned int grid_dim_x, unsigned int grid_dim_y,
    unsigned int grid_dim_z, unsigned int block_dim_x, unsigned int block_dim_y,
    unsigned int block_dim_z, unsigned int shared_memory_bytes, MUstream stream,
    void** kernel_parameters, void** extra) {
  ++g_state->launch_calls;
  g_state->last_launch = FakeApiState::LaunchArguments{
      function,    grid_dim_x,        grid_dim_y,  grid_dim_z,
      block_dim_x, block_dim_y,       block_dim_z, shared_memory_bytes,
      stream,      kernel_parameters, extra};
  return g_state->launch_result;
}

MUresult MUSAAPI FakeMuMemsetD32Async(MUdeviceptr destination,
                                      unsigned int value, size_t count,
                                      MUstream stream) {
  ++g_state->memset_calls;
  g_state->last_memset_destination = destination;
  g_state->last_memset_value = value;
  g_state->last_memset_count = count;
  g_state->last_memset_stream = stream;
  return g_state->memset_result;
}

void InstallBootstrap(FakeSymbolLoader& loader) {
  loader.Add("muInit", &FakeMuInit);
  loader.Add("muGetErrorName", &FakeMuGetErrorName);
  loader.Add("muGetErrorString", &FakeMuGetErrorString);
}

void InstallRequiredDlsymSymbols(FakeSymbolLoader& loader) {
  loader.Add("muDriverGetVersion", &FakeMuDriverGetVersion);
  loader.Add("muDeviceGetCount", &FakeMuDeviceGetCount);
  loader.Add("muDeviceGet", &FakeMuDeviceGet);
  loader.Add("muDeviceCanAccessPeer", &FakeMuDeviceCanAccessPeer);
  loader.Add("muDeviceGetP2PAttribute", &FakeMuDeviceGetP2PAttribute);
  loader.Add("muDevicePrimaryCtxRetain", &FakeMuDevicePrimaryCtxRetain);
  loader.Add("muDevicePrimaryCtxRelease_v2", &FakeMuDevicePrimaryCtxRelease);
  loader.Add("muDevicePrimaryCtxGetState", &FakeMuDevicePrimaryCtxGetState);
  loader.Add("muDevicePrimaryCtxSetFlags_v2", &FakeMuDevicePrimaryCtxSetFlags);
  loader.Add("muCtxSetCurrent", &FakeMuCtxSetCurrent);
  loader.Add("muCtxGetCurrent", &FakeMuCtxGetCurrent);
  loader.Add("muCtxGetDevice", &FakeMuCtxGetDevice);
  loader.Add("muCtxSynchronize", &FakeMuCtxSynchronize);
  loader.Add("muCtxEnablePeerAccess", &FakeMuCtxEnablePeerAccess);
  loader.Add("muPointerGetAttribute", &FakeMuPointerGetAttribute);
  loader.Add("muMemcpyPeerAsync", &FakeMuMemcpyPeerAsync);
  loader.Add("muModuleLoadData", &FakeMuModuleLoadData);
  loader.Add("muModuleUnload", &FakeMuModuleUnload);
  loader.Add("muModuleGetFunction", &FakeMuModuleGetFunction);
  loader.Add("muModuleGetGlobal_v2", &FakeMuModuleGetGlobal);
  loader.Add("muFuncGetAttribute", &FakeMuFuncGetAttribute);
  loader.Add("muOccupancyMaxActiveBlocksPerMultiprocessor",
             &FakeMuOccupancyMaxActiveBlocksPerMultiprocessor);
  loader.Add("muLaunchKernel", &FakeMuLaunchKernel);
  loader.Add("muMemsetD32Async", &FakeMuMemsetD32Async);
}

void InstallGpaSymbols(FakeApiState& state) {
  state.gpa_symbols["muDriverGetVersion"] =
      reinterpret_cast<void*>(&FakeMuDriverGetVersion);
  state.gpa_symbols["muDeviceGetCount"] =
      reinterpret_cast<void*>(&FakeMuDeviceGetCount);
  state.gpa_symbols["muDeviceGet"] = reinterpret_cast<void*>(&FakeMuDeviceGet);
  state.gpa_symbols["muDeviceCanAccessPeer"] =
      reinterpret_cast<void*>(&FakeMuDeviceCanAccessPeer);
  state.gpa_symbols["muDeviceGetP2PAttribute"] =
      reinterpret_cast<void*>(&FakeMuDeviceGetP2PAttribute);
  state.gpa_symbols["muDevicePrimaryCtxRetain"] =
      reinterpret_cast<void*>(&FakeMuDevicePrimaryCtxRetain);
  state.gpa_symbols["muDevicePrimaryCtxRelease"] =
      reinterpret_cast<void*>(&FakeMuDevicePrimaryCtxRelease);
  state.gpa_symbols["muDevicePrimaryCtxGetState"] =
      reinterpret_cast<void*>(&FakeMuDevicePrimaryCtxGetState);
  state.gpa_symbols["muDevicePrimaryCtxSetFlags"] =
      reinterpret_cast<void*>(&FakeMuDevicePrimaryCtxSetFlags);
  state.gpa_symbols["muCtxSetCurrent"] =
      reinterpret_cast<void*>(&FakeMuCtxSetCurrent);
  state.gpa_symbols["muCtxGetCurrent"] =
      reinterpret_cast<void*>(&FakeMuCtxGetCurrent);
  state.gpa_symbols["muCtxGetDevice"] =
      reinterpret_cast<void*>(&FakeMuCtxGetDevice);
  state.gpa_symbols["muCtxSynchronize"] =
      reinterpret_cast<void*>(&FakeMuCtxSynchronize);
  state.gpa_symbols["muCtxEnablePeerAccess"] =
      reinterpret_cast<void*>(&FakeMuCtxEnablePeerAccess);
  state.gpa_symbols["muPointerGetAttribute"] =
      reinterpret_cast<void*>(&FakeMuPointerGetAttribute);
  state.gpa_symbols["muMemcpyPeerAsync"] =
      reinterpret_cast<void*>(&FakeMuMemcpyPeerAsync);
  state.gpa_symbols["muModuleLoadData"] =
      reinterpret_cast<void*>(&FakeMuModuleLoadData);
  state.gpa_symbols["muModuleUnload"] =
      reinterpret_cast<void*>(&FakeMuModuleUnload);
  state.gpa_symbols["muModuleGetFunction"] =
      reinterpret_cast<void*>(&FakeMuModuleGetFunction);
  state.gpa_symbols["muModuleGetGlobal"] =
      reinterpret_cast<void*>(&FakeMuModuleGetGlobal);
  state.gpa_symbols["muFuncGetAttribute"] =
      reinterpret_cast<void*>(&FakeMuFuncGetAttribute);
  state.gpa_symbols["muOccupancyMaxActiveBlocksPerMultiprocessor"] =
      reinterpret_cast<void*>(&FakeMuOccupancyMaxActiveBlocksPerMultiprocessor);
  state.gpa_symbols["muLaunchKernel"] =
      reinterpret_cast<void*>(&FakeMuLaunchKernel);
  state.gpa_symbols["muMemsetD32Async"] =
      reinterpret_cast<void*>(&FakeMuMemsetD32Async);
}

std::unique_ptr<FakeSymbolLoader> CompleteLoader() {
  auto loader = std::make_unique<FakeSymbolLoader>();
  InstallBootstrap(*loader);
  InstallRequiredDlsymSymbols(*loader);
  return loader;
}

class MusaDriverTest : public ::testing::Test {
 protected:
  void SetUp() override { g_state = &state_; }
  void TearDown() override { g_state = nullptr; }

  FakeApiState state_;
};

TEST_F(MusaDriverTest, CompleteTableWithoutGetProcAddress) {
  auto loader = CompleteLoader();
  FakeSymbolLoader* loader_ptr = loader.get();
  MusaDriver driver(std::move(loader));

  EXPECT_TRUE(driver.Init().ok());
  absl::StatusOr<int> version = driver.DriverVersion();
  ASSERT_TRUE(version.ok());
  EXPECT_EQ(*version, 10504);
  absl::StatusOr<int> count = driver.DeviceCount();
  ASSERT_TRUE(count.ok());
  EXPECT_EQ(*count, 2);
  absl::StatusOr<MUdevice> device = driver.Device(0);
  ASSERT_TRUE(device.ok());
  EXPECT_EQ(*device, 10);
  absl::StatusOr<MusaPeerAccessInfo> peer = driver.PeerAccessInfo(10, 11);
  ASSERT_TRUE(peer.ok()) << peer.status();
  EXPECT_EQ(*peer, (MusaPeerAccessInfo{
                       .can_access_peer = true,
                       .link_attributes_available = true,
                       .performance_rank = 7,
                       .native_atomic_supported = true,
                       .musa_array_access_supported = false,
                       .mtlink_port_count = 2,
                   }));
  EXPECT_EQ(state_.last_peer_source, 10);
  EXPECT_EQ(state_.last_peer_target, 11);
  MUcontext peer_context = reinterpret_cast<MUcontext>(0x8888);
  EXPECT_TRUE(driver.EnablePeerAccess(peer_context).ok());
  EXPECT_EQ(state_.last_enabled_peer_context, peer_context);
  EXPECT_EQ(state_.last_peer_flags, MU_PEERACCESS_DEFAULT);
  absl::StatusOr<MUcontext> retained = driver.RetainPrimaryContext(10);
  ASSERT_TRUE(retained.ok());
  EXPECT_EQ(*retained, state_.retained_context);
  EXPECT_TRUE(driver.SetPrimaryContextFlags(10, 13).ok());
  absl::StatusOr<MusaPrimaryContextState> primary_state =
      driver.PrimaryContextState(10);
  ASSERT_TRUE(primary_state.ok());
  EXPECT_EQ(primary_state->flags, 13);
  EXPECT_TRUE(driver.SetCurrentContext(state_.retained_context).ok());
  absl::StatusOr<MUcontext> current = driver.CurrentContext();
  ASSERT_TRUE(current.ok());
  EXPECT_EQ(*current, state_.retained_context);
  absl::StatusOr<MUdevice> current_device = driver.CurrentDevice();
  ASSERT_TRUE(current_device.ok());
  EXPECT_EQ(*current_device, 11);
  EXPECT_TRUE(driver.SynchronizeContext().ok());
  const uint8_t image[] = {0x7f, 'E', 'L', 'F'};
  absl::StatusOr<MUmodule> module = driver.LoadModuleData(image);
  ASSERT_TRUE(module.ok());
  EXPECT_EQ(*module, state_.module);
  EXPECT_EQ(state_.last_module_image, image);
  EXPECT_TRUE(driver.UnloadModule(*module).ok());
  EXPECT_EQ(state_.last_unloaded_module, state_.module);
  absl::StatusOr<MUfunction> function =
      driver.GetModuleFunction(*module, "vector_add");
  ASSERT_TRUE(function.ok());
  EXPECT_EQ(*function, state_.function);
  EXPECT_EQ(state_.last_lookup_module, state_.module);
  EXPECT_EQ(state_.last_lookup_name, "vector_add");
  absl::StatusOr<MusaModuleGlobal> global =
      driver.GetModuleGlobal(*module, "constant_data");
  ASSERT_TRUE(global.ok());
  EXPECT_EQ(global->address, state_.global_address);
  EXPECT_EQ(global->size, state_.global_size);
  absl::StatusOr<int> attribute =
      driver.FunctionAttribute(*function, MU_FUNC_ATTRIBUTE_NUM_REGS);
  ASSERT_TRUE(attribute.ok());
  EXPECT_EQ(*attribute, 32);
  EXPECT_EQ(state_.last_function_attribute, MU_FUNC_ATTRIBUTE_NUM_REGS);
  absl::StatusOr<int> occupancy =
      driver.MaxActiveBlocksPerMultiprocessor(*function, 256, 1024);
  ASSERT_TRUE(occupancy.ok());
  EXPECT_EQ(*occupancy, 3);
  EXPECT_EQ(state_.last_occupancy_block_size, 256);
  EXPECT_EQ(state_.last_occupancy_dynamic_shared_memory_bytes, 1024);
  void* parameters[] = {reinterpret_cast<void*>(0x1111)};
  void* extra[] = {reinterpret_cast<void*>(0x2222)};
  MUstream stream = reinterpret_cast<MUstream>(0x3333);
  EXPECT_TRUE(driver
                  .LaunchKernel(*function, 5, 6, 7, 8, 9, 10, 2048, stream,
                                parameters, extra)
                  .ok());
  EXPECT_EQ(state_.last_launch.function, state_.function);
  EXPECT_EQ(state_.last_launch.grid_dim_x, 5);
  EXPECT_EQ(state_.last_launch.grid_dim_y, 6);
  EXPECT_EQ(state_.last_launch.grid_dim_z, 7);
  EXPECT_EQ(state_.last_launch.block_dim_x, 8);
  EXPECT_EQ(state_.last_launch.block_dim_y, 9);
  EXPECT_EQ(state_.last_launch.block_dim_z, 10);
  EXPECT_EQ(state_.last_launch.shared_memory_bytes, 2048);
  EXPECT_EQ(state_.last_launch.stream, stream);
  EXPECT_EQ(state_.last_launch.kernel_parameters, parameters);
  EXPECT_EQ(state_.last_launch.extra, extra);
  EXPECT_TRUE(
      driver.MemsetD32Async(state_.global_address, 0xa5a5f00d, 17, stream)
          .ok());
  EXPECT_EQ(state_.last_memset_destination, state_.global_address);
  EXPECT_EQ(state_.last_memset_value, 0xa5a5f00d);
  EXPECT_EQ(state_.last_memset_count, 17);
  EXPECT_EQ(state_.last_memset_stream, stream);
  EXPECT_TRUE(driver.ReleasePrimaryContext(10).ok());
  EXPECT_EQ(state_.release_calls, 1);
  EXPECT_EQ(loader_ptr->load_calls, 1);
}

TEST_F(MusaDriverTest, PeerFactsAreDirectionalAndSkipInaccessibleAttributes) {
  auto loader = CompleteLoader();
  MusaDriver driver(std::move(loader));

  absl::StatusOr<MusaPeerAccessInfo> forward =
      driver.PeerAccessInfo(/*source=*/10, /*peer=*/11);
  ASSERT_TRUE(forward.ok()) << forward.status();
  EXPECT_TRUE(forward->can_access_peer);
  EXPECT_TRUE(forward->link_attributes_available);
  EXPECT_EQ(forward->performance_rank, 7);
  EXPECT_TRUE(forward->native_atomic_supported);
  EXPECT_FALSE(forward->musa_array_access_supported);
  EXPECT_EQ(forward->mtlink_port_count, 2);
  EXPECT_EQ(state_.last_peer_source, 10);
  EXPECT_EQ(state_.last_peer_target, 11);
  EXPECT_EQ(state_.queried_p2p_attributes,
            (std::vector<MUdevice_P2PAttribute>{
                MU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK,
                MU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED,
                MU_DEVICE_P2P_ATTRIBUTE_MUSA_ARRAY_ACCESS_SUPPORTED,
                MU_DEVICE_P2P_ATTRIBUTE_MTLINK_PORT_COUNT,
            }));

  state_.peer_can_access = 0;
  state_.queried_p2p_attributes.clear();
  absl::StatusOr<MusaPeerAccessInfo> reverse =
      driver.PeerAccessInfo(/*source=*/11, /*peer=*/10);
  ASSERT_TRUE(reverse.ok()) << reverse.status();
  EXPECT_EQ(*reverse, MusaPeerAccessInfo{});
  EXPECT_EQ(state_.last_peer_source, 11);
  EXPECT_EQ(state_.last_peer_target, 10);
  EXPECT_TRUE(state_.queried_p2p_attributes.empty());
}

TEST_F(MusaDriverTest, PeerQueriesRejectInvalidCapabilityAndDegradeTelemetry) {
  auto loader = CompleteLoader();
  MusaDriver driver(std::move(loader));

  EXPECT_EQ(driver.PeerAccessInfo(10, 10).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(state_.peer_can_access_calls, 0);

  state_.peer_can_access = 2;
  absl::Status non_boolean = driver.PeerAccessInfo(10, 11).status();
  EXPECT_EQ(non_boolean.code(), absl::StatusCode::kInternal);
  EXPECT_TRUE(absl::StrContains(non_boolean.message(), "non-boolean"));

  state_.peer_can_access = 1;
  state_.p2p_attributes[MU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK] = -1;
  absl::StatusOr<MusaPeerAccessInfo> negative =
      driver.PeerAccessInfo(10, 11);
  ASSERT_TRUE(negative.ok()) << negative.status();
  EXPECT_TRUE(negative->can_access_peer);
  EXPECT_FALSE(negative->link_attributes_available);
  EXPECT_EQ(negative->performance_rank, 0);
  EXPECT_EQ(negative->mtlink_port_count, 0);

  state_.p2p_attributes[MU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK] = 7;
  state_.p2p_attributes[MU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED] = 2;
  absl::StatusOr<MusaPeerAccessInfo> malformed_boolean =
      driver.PeerAccessInfo(10, 11);
  ASSERT_TRUE(malformed_boolean.ok()) << malformed_boolean.status();
  EXPECT_TRUE(malformed_boolean->can_access_peer);
  EXPECT_FALSE(malformed_boolean->link_attributes_available);
}

TEST_F(MusaDriverTest, PeerErrorsAreCanonicalAndEnableIsIdempotent) {
  auto loader = CompleteLoader();
  MusaDriver driver(std::move(loader));

  state_.peer_can_access_result = MUSA_ERROR_INVALID_DEVICE;
  absl::Status query = driver.PeerAccessInfo(10, 11).status();
  EXPECT_EQ(query.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(query.message(), "muDeviceCanAccessPeer"));

  state_.peer_can_access_result = MUSA_SUCCESS;
  state_.p2p_attribute_result = MUSA_ERROR_INVALID_VALUE;
  absl::StatusOr<MusaPeerAccessInfo> attribute =
      driver.PeerAccessInfo(10, 11);
  ASSERT_TRUE(attribute.ok()) << attribute.status();
  EXPECT_TRUE(attribute->can_access_peer);
  EXPECT_FALSE(attribute->link_attributes_available);

  EXPECT_EQ(driver.EnablePeerAccess(nullptr).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(state_.enable_peer_calls, 0);

  MUcontext source_context = reinterpret_cast<MUcontext>(0x1111);
  MUcontext peer_context = reinterpret_cast<MUcontext>(0x2222);
  ASSERT_TRUE(driver.SetCurrentContext(source_context).ok());
  ASSERT_TRUE(driver.EnablePeerAccess(peer_context).ok());
  EXPECT_EQ(state_.current_context, source_context);
  EXPECT_EQ(state_.last_enabled_peer_context, peer_context);
  EXPECT_EQ(state_.last_peer_flags, MU_PEERACCESS_DEFAULT);

  state_.enable_peer_result = MUSA_ERROR_PEER_ACCESS_ALREADY_ENABLED;
  EXPECT_TRUE(driver.EnablePeerAccess(peer_context).ok());
  EXPECT_EQ(state_.enable_peer_calls, 2);

  state_.enable_peer_result = MUSA_ERROR_PEER_ACCESS_UNSUPPORTED;
  absl::Status unsupported = driver.EnablePeerAccess(peer_context);
  EXPECT_EQ(unsupported.code(), absl::StatusCode::kUnimplemented);
  EXPECT_TRUE(absl::StrContains(unsupported.message(),
                                "muCtxEnablePeerAccess"));

  state_.enable_peer_result = MUSA_ERROR_TOO_MANY_PEERS;
  EXPECT_EQ(driver.EnablePeerAccess(peer_context).code(),
            absl::StatusCode::kResourceExhausted);
}

TEST_F(MusaDriverTest, PointerContextsAndPeerCopiesPreserveExactAbi) {
  auto loader = CompleteLoader();
  MusaDriver driver(std::move(loader));

  EXPECT_EQ(driver.ContextForPointer(0).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(state_.pointer_attribute_calls, 0);

  constexpr MUdeviceptr kDestination = 0xaaaabbbb;
  constexpr MUdeviceptr kSource = 0xccccdddd;
  absl::StatusOr<MUcontext> pointer_context =
      driver.ContextForPointer(kDestination);
  ASSERT_TRUE(pointer_context.ok()) << pointer_context.status();
  EXPECT_EQ(*pointer_context, state_.pointer_context);
  EXPECT_EQ(state_.last_pointer, kDestination);
  EXPECT_EQ(state_.last_pointer_attribute, MU_POINTER_ATTRIBUTE_CONTEXT);

  state_.pointer_context = nullptr;
  EXPECT_EQ(driver.ContextForPointer(kSource).status().code(),
            absl::StatusCode::kInternal);
  state_.pointer_attribute_result = MUSA_ERROR_INVALID_VALUE;
  EXPECT_EQ(driver.ContextForPointer(kSource).status().code(),
            absl::StatusCode::kInvalidArgument);
  state_.pointer_attribute_result = MUSA_SUCCESS;
  state_.pointer_context = reinterpret_cast<MUcontext>(0x3344);

  MUcontext destination_context = reinterpret_cast<MUcontext>(0x1111);
  MUcontext source_context = reinterpret_cast<MUcontext>(0x2222);
  MUstream stream = reinterpret_cast<MUstream>(0x3333);
  EXPECT_EQ(driver
                .MemcpyPeerAsync(0, destination_context, kSource,
                                 source_context, 4096, stream)
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(driver
                .MemcpyPeerAsync(kDestination, nullptr, kSource,
                                 source_context, 4096, stream)
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(state_.peer_copy_calls, 0);

  EXPECT_TRUE(driver
                  .MemcpyPeerAsync(kDestination, destination_context, kSource,
                                   source_context, 4096, stream)
                  .ok());
  EXPECT_EQ(state_.peer_copy_calls, 1);
  EXPECT_EQ(state_.last_copy_destination, kDestination);
  EXPECT_EQ(state_.last_copy_destination_context, destination_context);
  EXPECT_EQ(state_.last_copy_source, kSource);
  EXPECT_EQ(state_.last_copy_source_context, source_context);
  EXPECT_EQ(state_.last_copy_bytes, 4096);
  EXPECT_EQ(state_.last_copy_stream, stream);

  state_.peer_copy_result = MUSA_ERROR_PEER_ACCESS_NOT_ENABLED;
  absl::Status not_enabled = driver.MemcpyPeerAsync(
      kDestination, destination_context, kSource, source_context, 4096, stream);
  EXPECT_EQ(not_enabled.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_TRUE(absl::StrContains(not_enabled.message(), "muMemcpyPeerAsync"));
}

TEST_F(MusaDriverTest, FourArgumentGetProcAddress) {
  auto loader = std::make_unique<FakeSymbolLoader>();
  InstallBootstrap(*loader);
  loader->Add("muGetProcAddress", &FakeMuGetProcAddress);
  InstallGpaSymbols(state_);
  MusaDriver driver(std::move(loader));

  EXPECT_TRUE(driver.Init().ok());
  EXPECT_GT(state_.gpa_calls, 0);
  EXPECT_EQ(state_.last_gpa_version, MUSA_VERSION);
  EXPECT_EQ(state_.last_gpa_flags,
            static_cast<uint64_t>(MU_GET_PROC_ADDRESS_LEGACY_STREAM));
}

TEST_F(MusaDriverTest, GetProcAddressSuccessWithNullFallsBackToDlsym) {
  auto loader = CompleteLoader();
  loader->Add("muGetProcAddress", &FakeMuGetProcAddress);
  state_.gpa_success_null = true;
  MusaDriver driver(std::move(loader));

  EXPECT_TRUE(driver.Init().ok());
  EXPECT_GT(state_.gpa_calls, 0);
  ASSERT_TRUE(driver.DeviceCount().ok());
  EXPECT_EQ(*driver.DeviceCount(), 2);
}

TEST_F(MusaDriverTest, ErrorHelpersAreOptional) {
  auto loader = CompleteLoader();
  loader->Remove("muGetErrorName");
  loader->Remove("muGetErrorString");
  MusaDriver driver(std::move(loader));

  EXPECT_TRUE(driver.Init().ok());
  ASSERT_TRUE(driver.DeviceCount().ok());
  EXPECT_EQ(*driver.DeviceCount(), 2);
}

TEST_F(MusaDriverTest, UsesVersionedPrimaryContextAliases) {
  auto loader = CompleteLoader();
  FakeSymbolLoader* loader_ptr = loader.get();
  MusaDriver driver(std::move(loader));

  EXPECT_TRUE(driver.Init().ok());
  EXPECT_EQ(loader_ptr->ResolveCalls("muDevicePrimaryCtxRelease_v2"), 1);
  EXPECT_EQ(loader_ptr->ResolveCalls("muDevicePrimaryCtxSetFlags_v2"), 1);
  EXPECT_EQ(loader_ptr->ResolveCalls("muDevicePrimaryCtxRelease"), 0);
  EXPECT_EQ(loader_ptr->ResolveCalls("muDevicePrimaryCtxSetFlags"), 0);
  EXPECT_EQ(loader_ptr->ResolveCalls("muModuleGetGlobal_v2"), 1);
  EXPECT_EQ(loader_ptr->ResolveCalls("muModuleGetGlobal"), 0);
}

TEST_F(MusaDriverTest, PrimaryContextErrorsNameLogicalOperations) {
  auto loader = CompleteLoader();
  MusaDriver driver(std::move(loader));
  ASSERT_TRUE(driver.Init().ok());

  state_.release_result = MUSA_ERROR_INVALID_VALUE;
  absl::Status release = driver.ReleasePrimaryContext(10);
  EXPECT_TRUE(
      absl::StrContains(release.message(), "muDevicePrimaryCtxRelease failed"));
  EXPECT_FALSE(absl::StrContains(release.message(),
                                 "muDevicePrimaryCtxRelease_v2 failed"));

  state_.set_flags_result = MUSA_ERROR_INVALID_VALUE;
  absl::Status set_flags = driver.SetPrimaryContextFlags(10, 0);
  EXPECT_TRUE(absl::StrContains(set_flags.message(),
                                "muDevicePrimaryCtxSetFlags failed"));
  EXPECT_FALSE(absl::StrContains(set_flags.message(),
                                 "muDevicePrimaryCtxSetFlags_v2 failed"));
}

TEST_F(MusaDriverTest, MissingRequiredSymbolHasCompleteDiagnostic) {
  auto loader = CompleteLoader();
  loader->Remove("muCtxSynchronize");
  MusaDriver driver(std::move(loader));

  absl::Status status = driver.Init();
  EXPECT_EQ(status.code(), absl::StatusCode::kNotFound);
  EXPECT_TRUE(absl::StrContains(status.message(), "muCtxSynchronize"));
  EXPECT_TRUE(absl::StrContains(status.message(), "MUSA_VERSION="));
  EXPECT_TRUE(absl::StrContains(status.message(), "flags=1"));
  EXPECT_TRUE(absl::StrContains(status.message(), "dlsym"));
  EXPECT_TRUE(absl::StrContains(status.message(), "/fake/libmusa.so.1.5"));
}

TEST_F(MusaDriverTest, EveryMissingRequiredCapabilityIsNamed) {
  struct MissingCapability {
    const char* logical_name;
    std::vector<const char*> dlsym_aliases;
  };
  const std::vector<MissingCapability> capabilities = {
      {"muInit", {"muInit"}},
      {"muDriverGetVersion", {"muDriverGetVersion"}},
      {"muDeviceGetCount", {"muDeviceGetCount"}},
      {"muDeviceGet", {"muDeviceGet"}},
      {"muDeviceCanAccessPeer", {"muDeviceCanAccessPeer"}},
      {"muDeviceGetP2PAttribute", {"muDeviceGetP2PAttribute"}},
      {"muDevicePrimaryCtxRetain", {"muDevicePrimaryCtxRetain"}},
      {"muDevicePrimaryCtxRelease",
       {"muDevicePrimaryCtxRelease_v2", "muDevicePrimaryCtxRelease"}},
      {"muDevicePrimaryCtxGetState", {"muDevicePrimaryCtxGetState"}},
      {"muDevicePrimaryCtxSetFlags",
       {"muDevicePrimaryCtxSetFlags_v2", "muDevicePrimaryCtxSetFlags"}},
      {"muCtxSetCurrent", {"muCtxSetCurrent"}},
      {"muCtxGetCurrent", {"muCtxGetCurrent"}},
      {"muCtxGetDevice", {"muCtxGetDevice"}},
      {"muCtxSynchronize", {"muCtxSynchronize"}},
      {"muCtxEnablePeerAccess", {"muCtxEnablePeerAccess"}},
      {"muPointerGetAttribute", {"muPointerGetAttribute"}},
      {"muMemcpyPeerAsync", {"muMemcpyPeerAsync"}},
      {"muModuleLoadData", {"muModuleLoadData"}},
      {"muModuleUnload", {"muModuleUnload"}},
      {"muModuleGetFunction", {"muModuleGetFunction"}},
      {"muModuleGetGlobal", {"muModuleGetGlobal_v2"}},
      {"muFuncGetAttribute", {"muFuncGetAttribute"}},
      {"muOccupancyMaxActiveBlocksPerMultiprocessor",
       {"muOccupancyMaxActiveBlocksPerMultiprocessor"}},
      {"muLaunchKernel", {"muLaunchKernel"}},
      {"muMemsetD32Async", {"muMemsetD32Async"}},
  };

  for (const MissingCapability& capability : capabilities) {
    SCOPED_TRACE(capability.logical_name);
    auto loader = CompleteLoader();
    for (const char* alias : capability.dlsym_aliases) loader->Remove(alias);
    MusaDriver driver(std::move(loader));

    absl::Status status = driver.Init();
    EXPECT_EQ(status.code(), absl::StatusCode::kNotFound);
    EXPECT_TRUE(absl::StrContains(status.message(), capability.logical_name))
        << status;
  }
}

TEST_F(MusaDriverTest, InitializationFailureIsCachedExactlyOnce) {
  auto loader = CompleteLoader();
  FakeSymbolLoader* loader_ptr = loader.get();
  state_.init_result = MUSA_ERROR_NOT_INITIALIZED;
  MusaDriver driver(std::move(loader));

  absl::Status first = driver.Init();
  absl::Status second = driver.Init();
  absl::Status count = driver.DeviceCount().status();
  EXPECT_EQ(first, second);
  EXPECT_EQ(first, count);
  EXPECT_EQ(first.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(state_.init_calls, 1);
  EXPECT_EQ(loader_ptr->load_calls, 1);
}

TEST_F(MusaDriverTest, NullRetainedContextIsRejected) {
  auto loader = CompleteLoader();
  state_.retained_context = nullptr;
  MusaDriver driver(std::move(loader));

  absl::Status status = driver.RetainPrimaryContext(10).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_TRUE(absl::StrContains(status.message(), "null context"));
}

TEST_F(MusaDriverTest, ModuleOperationsRejectNullInputsAndResults) {
  auto loader = CompleteLoader();
  MusaDriver driver(std::move(loader));

  EXPECT_EQ(driver.LoadModuleData(nullptr).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(driver.UnloadModule(nullptr).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(state_.module_load_calls, 0);
  EXPECT_EQ(state_.module_unload_calls, 0);

  const uint8_t image[] = {0x7f, 'E', 'L', 'F'};
  state_.module = nullptr;
  absl::Status status = driver.LoadModuleData(image).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_TRUE(absl::StrContains(status.message(), "null module"));
  EXPECT_EQ(state_.module_load_calls, 1);
}

TEST_F(MusaDriverTest, ModuleErrorsUseCanonicalDriverStatus) {
  auto loader = CompleteLoader();
  MusaDriver driver(std::move(loader));
  const uint8_t image[] = {0x7f, 'E', 'L', 'F'};

  state_.module_load_result = MUSA_ERROR_INVALID_IMAGE;
  absl::Status load = driver.LoadModuleData(image).status();
  EXPECT_EQ(load.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(load.message(), "muModuleLoadData failed"));

  state_.module_unload_result = MUSA_ERROR_INVALID_HANDLE;
  absl::Status unload = driver.UnloadModule(state_.module);
  EXPECT_EQ(unload.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(unload.message(), "muModuleUnload failed"));
}

TEST_F(MusaDriverTest, ModuleLookupsRejectInvalidInputsAndNullOutputs) {
  auto loader = CompleteLoader();
  MusaDriver driver(std::move(loader));

  EXPECT_EQ(driver.GetModuleFunction(nullptr, "kernel").status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(driver.GetModuleFunction(state_.module, nullptr).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(driver.GetModuleFunction(state_.module, "").status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(driver.GetModuleGlobal(nullptr, "global").status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(driver.GetModuleGlobal(state_.module, nullptr).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(driver.GetModuleGlobal(state_.module, "").status().code(),
            absl::StatusCode::kInvalidArgument);

  state_.function = nullptr;
  absl::Status function =
      driver.GetModuleFunction(state_.module, "kernel").status();
  EXPECT_EQ(function.code(), absl::StatusCode::kInternal);
  EXPECT_TRUE(absl::StrContains(function.message(), "null function"));

  state_.global_address = 0;
  absl::Status address =
      driver.GetModuleGlobal(state_.module, "global").status();
  EXPECT_EQ(address.code(), absl::StatusCode::kInternal);
  EXPECT_TRUE(absl::StrContains(address.message(), "null address"));

  state_.global_address = static_cast<MUdeviceptr>(0xabc000);
  state_.global_size = 0;
  absl::Status size = driver.GetModuleGlobal(state_.module, "global").status();
  EXPECT_EQ(size.code(), absl::StatusCode::kInternal);
  EXPECT_TRUE(absl::StrContains(size.message(), "zero-byte symbol"));
}

TEST_F(MusaDriverTest, KernelQueriesAndLaunchValidateInputsAndOutputs) {
  auto loader = CompleteLoader();
  MusaDriver driver(std::move(loader));

  EXPECT_EQ(
      driver.FunctionAttribute(nullptr, MU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
          .status()
          .code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      driver.MaxActiveBlocksPerMultiprocessor(nullptr, 1, 0).status().code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(driver.MaxActiveBlocksPerMultiprocessor(state_.function, 0, 0)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);

  state_.occupancy_blocks = -1;
  absl::Status occupancy =
      driver.MaxActiveBlocksPerMultiprocessor(state_.function, 128, 0).status();
  EXPECT_EQ(occupancy.code(), absl::StatusCode::kInternal);
  EXPECT_TRUE(absl::StrContains(occupancy.message(), "negative block count"));

  EXPECT_EQ(
      driver
          .LaunchKernel(nullptr, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr)
          .code(),
      absl::StatusCode::kInvalidArgument);
  const std::vector<std::vector<unsigned int>> dimensions = {
      {0, 1, 1, 1, 1, 1}, {1, 0, 1, 1, 1, 1}, {1, 1, 0, 1, 1, 1},
      {1, 1, 1, 0, 1, 1}, {1, 1, 1, 1, 0, 1}, {1, 1, 1, 1, 1, 0},
  };
  for (const std::vector<unsigned int>& dims : dimensions) {
    EXPECT_EQ(
        driver
            .LaunchKernel(state_.function, dims[0], dims[1], dims[2], dims[3],
                          dims[4], dims[5], 0, nullptr, nullptr, nullptr)
            .code(),
        absl::StatusCode::kInvalidArgument);
  }
  EXPECT_EQ(state_.launch_calls, 0);

  EXPECT_EQ(driver.MemsetD32Async(0, 0, 1, nullptr).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(state_.memset_calls, 0);
}

TEST_F(MusaDriverTest, LookupQueryLaunchAndMemsetErrorsAreCanonical) {
  auto loader = CompleteLoader();
  MusaDriver driver(std::move(loader));

  state_.module_get_function_result = MUSA_ERROR_NOT_FOUND;
  absl::Status function =
      driver.GetModuleFunction(state_.module, "missing_kernel").status();
  EXPECT_EQ(function.code(), absl::StatusCode::kNotFound);
  EXPECT_TRUE(
      absl::StrContains(function.message(), "muModuleGetFunction failed"));

  state_.module_get_global_result = MUSA_ERROR_NOT_FOUND;
  absl::Status global =
      driver.GetModuleGlobal(state_.module, "missing_global").status();
  EXPECT_EQ(global.code(), absl::StatusCode::kNotFound);
  EXPECT_TRUE(absl::StrContains(global.message(), "muModuleGetGlobal failed"));

  state_.function_attribute_result = MUSA_ERROR_INVALID_HANDLE;
  absl::Status attribute =
      driver
          .FunctionAttribute(state_.function,
                             MU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
          .status();
  EXPECT_EQ(attribute.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(
      absl::StrContains(attribute.message(), "muFuncGetAttribute failed"));

  state_.occupancy_result = MUSA_ERROR_INVALID_VALUE;
  absl::Status occupancy =
      driver.MaxActiveBlocksPerMultiprocessor(state_.function, 128, 4096)
          .status();
  EXPECT_EQ(occupancy.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(
      absl::StrContains(occupancy.message(),
                        "muOccupancyMaxActiveBlocksPerMultiprocessor failed"));

  state_.launch_result = MUSA_ERROR_LAUNCH_OUT_OF_RESOURCES;
  absl::Status launch = driver.LaunchKernel(state_.function, 1, 1, 1, 128, 1, 1,
                                            0, nullptr, nullptr, nullptr);
  EXPECT_EQ(launch.code(), absl::StatusCode::kResourceExhausted);
  EXPECT_TRUE(absl::StrContains(launch.message(), "muLaunchKernel failed"));

  state_.memset_result = MUSA_ERROR_INVALID_VALUE;
  absl::Status memset = driver.MemsetD32Async(
      static_cast<MUdeviceptr>(0xabc000), 0xff00ff00, 5, nullptr);
  EXPECT_EQ(memset.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(memset.message(), "muMemsetD32Async failed"));
}

TEST_F(MusaDriverTest, DriverErrorsHaveCanonicalCodeNameAndDescription) {
  absl::Status out_of_memory =
      DriverToStatus(MUSA_ERROR_OUT_OF_MEMORY, "test allocation",
                     &FakeMuGetErrorName, &FakeMuGetErrorString);
  EXPECT_EQ(out_of_memory.code(), absl::StatusCode::kResourceExhausted);
  EXPECT_TRUE(
      absl::StrContains(out_of_memory.message(), "MUSA_ERROR_OUT_OF_MEMORY"));
  EXPECT_TRUE(absl::StrContains(out_of_memory.message(), "out of memory"));

  absl::Status invalid =
      DriverToStatus(MUSA_ERROR_INVALID_VALUE, "test argument",
                     &FakeMuGetErrorName, &FakeMuGetErrorString);
  EXPECT_EQ(invalid.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(invalid.message(), "invalid value"));

  EXPECT_EQ(DriverToStatus(MUSA_ERROR_NO_DEVICE, "device discovery",
                           &FakeMuGetErrorName, &FakeMuGetErrorString)
                .code(),
            absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(
      DriverToStatus(MUSA_ERROR_SYSTEM_DRIVER_MISMATCH, "driver initialization",
                     &FakeMuGetErrorName, &FakeMuGetErrorString)
          .code(),
      absl::StatusCode::kFailedPrecondition);
  EXPECT_TRUE(
      absl::StrContains(out_of_memory.message(), "test allocation failed:"));
}

TEST_F(MusaDriverTest, ConcurrentCallsDoNotSerializeOnLoaderLock) {
  auto loader = CompleteLoader();
  MusaDriver driver(std::move(loader));
  ASSERT_TRUE(driver.Init().ok());
  state_.block_device_count = true;

  absl::StatusOr<int> first = absl::UnknownError("first call not started");
  absl::StatusOr<int> second = absl::UnknownError("second call not started");
  std::thread first_thread([&] { first = driver.DeviceCount(); });
  std::thread second_thread([&] { second = driver.DeviceCount(); });
  first_thread.join();
  second_thread.join();

  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(second.ok());
  EXPECT_EQ(*first, 2);
  EXPECT_EQ(*second, 2);
  EXPECT_EQ(state_.max_active_device_count_calls, 2);
}

}  // namespace
}  // namespace stream_executor::musa
