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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_MODULE_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_MODULE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "musa.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/musa/musa_context.h"
#include "xla/stream_executor/musa/musa_driver.h"
#include "xla/stream_executor/musa/musa_mubin.h"

namespace stream_executor::musa {

// Device binaries retain an explicit kind even though MUBIN is currently the
// only binary format accepted by the MUSA backend. This prevents a MUBIN from
// becoming cache-compatible with a CUDA CUBIN or a future textual format.
enum class MusaBinaryKind : uint8_t { kMubin = 1 };

// Owns a loaded MUSA module, its validated MUBIN bytes, and the primary context
// in which the module was loaded. The caller-provided MusaDriver must outlive
// the module (production uses the process-wide driver singleton).
class MusaModule final {
 public:
  static absl::StatusOr<std::shared_ptr<MusaModule>> Load(
      MusaDriver* driver, std::shared_ptr<MusaContext> context,
      absl::Span<const uint8_t> mubin, absl::string_view architecture,
      uint32_t binary_abi_version = kMubinLoaderAbiVersion);

  ~MusaModule();

  MusaModule(const MusaModule&) = delete;
  MusaModule& operator=(const MusaModule&) = delete;

  MUmodule module() const { return module_; }
  MusaDriver* driver() const { return driver_; }
  MusaContext* context() const { return context_.get(); }
  MUcontext native_context() const { return context_->context(); }
  MUdevice device() const { return context_->device(); }
  int device_ordinal() const { return context_->device_ordinal(); }
  MusaBinaryKind binary_kind() const { return MusaBinaryKind::kMubin; }
  absl::Span<const uint8_t> bytes() const { return *bytes_; }
  const MusaMubinMetadata& metadata() const { return metadata_; }
  const std::string& architecture() const { return architecture_; }
  uint32_t binary_abi_version() const { return binary_abi_version_; }

  absl::StatusOr<MUfunction> GetFunction(absl::string_view name) const;
  absl::StatusOr<MusaModuleGlobal> GetGlobal(absl::string_view name) const;

 private:
  friend class MusaModuleCache;

  static absl::StatusOr<std::shared_ptr<MusaModule>> LoadValidated(
      MusaDriver* driver, std::shared_ptr<MusaContext> context,
      std::shared_ptr<const std::vector<uint8_t>> bytes,
      MusaMubinMetadata metadata, std::string architecture,
      uint32_t binary_abi_version);

  MusaModule(MusaDriver* driver, std::shared_ptr<MusaContext> context,
             std::shared_ptr<const std::vector<uint8_t>> bytes,
             MusaMubinMetadata metadata, std::string architecture,
             uint32_t binary_abi_version, MUmodule module);

  MusaDriver* const driver_;
  const std::shared_ptr<MusaContext> context_;
  const std::shared_ptr<const std::vector<uint8_t>> bytes_;
  const MusaMubinMetadata metadata_;
  const std::string architecture_;
  const uint32_t binary_abi_version_;
  MUmodule module_;

  mutable absl::Mutex symbols_mutex_;
  mutable absl::flat_hash_map<std::string, MUfunction> functions_
      ABSL_GUARDED_BY(symbols_mutex_);
  mutable absl::flat_hash_map<std::string, MusaModuleGlobal> globals_
      ABSL_GUARDED_BY(symbols_mutex_);
};

// Thread-safe, weak module cache for one MUSA context and architecture.
//
// GetOrLoadModule returns shared ownership directly. AcquireModuleHandle adds a
// counted StreamExecutor client reference; each successful acquisition must be
// paired with ReleaseModuleHandle. Native loads happen outside the cache mutex,
// and a concurrent speculative loser is unloaded only after the mutex is
// released.
class MusaModuleCache final {
 public:
  MusaModuleCache(MusaDriver* driver, std::shared_ptr<MusaContext> context,
                  std::string architecture,
                  uint32_t binary_abi_version = kMubinLoaderAbiVersion);
  ~MusaModuleCache();

  MusaModuleCache(const MusaModuleCache&) = delete;
  MusaModuleCache& operator=(const MusaModuleCache&) = delete;

  absl::StatusOr<std::shared_ptr<MusaModule>> GetOrLoadModule(
      absl::Span<const uint8_t> mubin);
  absl::StatusOr<ModuleHandle> AcquireModuleHandle(
      absl::Span<const uint8_t> mubin);
  absl::StatusOr<std::shared_ptr<MusaModule>> LookupModule(
      ModuleHandle handle) const;
  bool ReleaseModuleHandle(ModuleHandle handle);

  // True when neither handle clients nor direct shared module owners remain.
  bool IsQuiescent() const;

 private:
  struct CacheState;

  MusaDriver* const driver_;
  const std::shared_ptr<MusaContext> context_;
  const std::string architecture_;
  const uint32_t binary_abi_version_;
  const std::unique_ptr<CacheState> state_;
};

}  // namespace stream_executor::musa

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_MODULE_H_
