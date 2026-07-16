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

#include <cstddef>
#include <cstdint>
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

// Address and size of a named global owned by a loaded MUSA module. The
// address remains valid only while the module remains loaded.
struct MusaModuleGlobal {
  MUdeviceptr address;
  size_t size;
};

// Typed, dynamically loaded subset of the MUSA driver API used by
// StreamExecutor. Initialization is attempted exactly once; both success and
// failure are cached. The resolved function table is immutable and ordinary
// calls never take the loader's initialization lock.
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
  virtual absl::StatusOr<MUfunction> GetModuleFunction(MUmodule module,
                                                       const char* name);
  virtual absl::StatusOr<MusaModuleGlobal> GetModuleGlobal(MUmodule module,
                                                           const char* name);

  virtual absl::StatusOr<int> FunctionAttribute(MUfunction function,
                                                MUfunction_attribute attribute);
  virtual absl::StatusOr<int> MaxActiveBlocksPerMultiprocessor(
      MUfunction function, int block_size, size_t dynamic_shared_memory_bytes);

  virtual absl::Status LaunchKernel(
      MUfunction function, unsigned int grid_dim_x, unsigned int grid_dim_y,
      unsigned int grid_dim_z, unsigned int block_dim_x,
      unsigned int block_dim_y, unsigned int block_dim_z,
      unsigned int shared_memory_bytes, MUstream stream,
      void** kernel_parameters, void** extra);

  // `count` is the number of 32-bit elements, matching the native driver ABI;
  // callers with a byte count must validate divisibility by four first.
  virtual absl::Status MemsetD32Async(MUdeviceptr destination, uint32_t value,
                                      size_t count, MUstream stream);

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
