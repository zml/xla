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

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
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

}  // namespace

struct MusaDriver::Api {
  MuGetErrorNameFn get_error_name = nullptr;
  MuGetErrorStringFn get_error_string = nullptr;
  MuDriverGetVersionFn driver_get_version = nullptr;
  MuDeviceGetCountFn device_get_count = nullptr;
  MuDeviceGetFn device_get = nullptr;
  MuDevicePrimaryCtxRetainFn primary_context_retain = nullptr;
  MuDevicePrimaryCtxReleaseV2Fn primary_context_release = nullptr;
  MuDevicePrimaryCtxGetStateFn primary_context_get_state = nullptr;
  MuDevicePrimaryCtxSetFlagsV2Fn primary_context_set_flags = nullptr;
  MuCtxSetCurrentFn context_set_current = nullptr;
  MuCtxGetCurrentFn context_get_current = nullptr;
  MuCtxGetDeviceFn context_get_device = nullptr;
  MuCtxSynchronizeFn context_synchronize = nullptr;
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

}  // namespace stream_executor::musa
