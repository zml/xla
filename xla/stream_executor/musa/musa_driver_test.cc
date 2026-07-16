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

void InstallBootstrap(FakeSymbolLoader& loader) {
  loader.Add("muInit", &FakeMuInit);
  loader.Add("muGetErrorName", &FakeMuGetErrorName);
  loader.Add("muGetErrorString", &FakeMuGetErrorString);
}

void InstallRequiredDlsymSymbols(FakeSymbolLoader& loader) {
  loader.Add("muDriverGetVersion", &FakeMuDriverGetVersion);
  loader.Add("muDeviceGetCount", &FakeMuDeviceGetCount);
  loader.Add("muDeviceGet", &FakeMuDeviceGet);
  loader.Add("muDevicePrimaryCtxRetain", &FakeMuDevicePrimaryCtxRetain);
  loader.Add("muDevicePrimaryCtxRelease_v2", &FakeMuDevicePrimaryCtxRelease);
  loader.Add("muDevicePrimaryCtxGetState", &FakeMuDevicePrimaryCtxGetState);
  loader.Add("muDevicePrimaryCtxSetFlags_v2", &FakeMuDevicePrimaryCtxSetFlags);
  loader.Add("muCtxSetCurrent", &FakeMuCtxSetCurrent);
  loader.Add("muCtxGetCurrent", &FakeMuCtxGetCurrent);
  loader.Add("muCtxGetDevice", &FakeMuCtxGetDevice);
  loader.Add("muCtxSynchronize", &FakeMuCtxSynchronize);
  loader.Add("muModuleLoadData", &FakeMuModuleLoadData);
  loader.Add("muModuleUnload", &FakeMuModuleUnload);
}

void InstallGpaSymbols(FakeApiState& state) {
  state.gpa_symbols["muDriverGetVersion"] =
      reinterpret_cast<void*>(&FakeMuDriverGetVersion);
  state.gpa_symbols["muDeviceGetCount"] =
      reinterpret_cast<void*>(&FakeMuDeviceGetCount);
  state.gpa_symbols["muDeviceGet"] = reinterpret_cast<void*>(&FakeMuDeviceGet);
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
  state.gpa_symbols["muModuleLoadData"] =
      reinterpret_cast<void*>(&FakeMuModuleLoadData);
  state.gpa_symbols["muModuleUnload"] =
      reinterpret_cast<void*>(&FakeMuModuleUnload);
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
  EXPECT_TRUE(driver.ReleasePrimaryContext(10).ok());
  EXPECT_EQ(state_.release_calls, 1);
  EXPECT_EQ(loader_ptr->load_calls, 1);
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
      {"muModuleLoadData", {"muModuleLoadData"}},
      {"muModuleUnload", {"muModuleUnload"}},
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
