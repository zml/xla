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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_DRIVER_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_DRIVER_H_

#include <memory>
#include <mutex>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "musa.h"
#include "xla/stream_executor/musa/musa_dso_loader.h"

namespace stream_executor::musa {

struct MusaPrimaryContextState {
  unsigned int flags;
  bool active;
};

// Typed, dynamically loaded subset of the MUSA driver API needed to establish
// StreamExecutor context ownership. Initialization is attempted exactly once;
// both success and failure are cached. The resolved function table is immutable
// and ordinary calls never take the loader's initialization lock.
class MusaDriver {
 public:
  MusaDriver();
  explicit MusaDriver(
      std::unique_ptr<internal::MusaSymbolLoader> symbol_loader);
  virtual ~MusaDriver();

  MusaDriver(const MusaDriver&) = delete;
  MusaDriver& operator=(const MusaDriver&) = delete;

  static MusaDriver& Instance();

  virtual absl::Status Init();
  virtual absl::StatusOr<int> DriverVersion();
  virtual absl::StatusOr<int> DeviceCount();
  virtual absl::StatusOr<MUdevice> Device(int ordinal);

  virtual absl::StatusOr<MUcontext> RetainPrimaryContext(MUdevice device);
  virtual absl::Status ReleasePrimaryContext(MUdevice device);
  virtual absl::Status SetPrimaryContextFlags(MUdevice device,
                                              unsigned int flags);
  virtual absl::StatusOr<MusaPrimaryContextState> PrimaryContextState(
      MUdevice device);

  virtual absl::Status SetCurrentContext(MUcontext context);
  virtual absl::StatusOr<MUcontext> CurrentContext();
  virtual absl::StatusOr<MUdevice> CurrentDevice();
  virtual absl::Status SynchronizeContext();

  virtual absl::StatusOr<MUmodule> LoadModuleData(const void* image);
  virtual absl::Status UnloadModule(MUmodule module);

 private:
  struct Api;

  absl::Status Initialize();
  absl::Status ResultStatus(MUresult result, const char* expression) const;

  std::unique_ptr<internal::MusaSymbolLoader> symbol_loader_;
  std::once_flag init_once_;
  absl::Status init_status_ =
      absl::UnknownError("MUSA driver initialization has not run");
  std::unique_ptr<const Api> api_;
};

}  // namespace stream_executor::musa

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_DRIVER_H_
