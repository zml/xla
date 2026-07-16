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

#include "xla/stream_executor/musa/musa_module.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/stream_executor/gpu/scoped_activate_context.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/musa/musa_context.h"
#include "xla/stream_executor/musa/musa_driver.h"
#include "xla/stream_executor/musa/musa_mubin.h"
#include "tsl/platform/fingerprint.h"

namespace stream_executor::musa {
namespace {

absl::Status ValidateLoadInputs(MusaDriver* driver,
                                const std::shared_ptr<MusaContext>& context,
                                absl::string_view architecture,
                                uint32_t binary_abi_version) {
  if (driver == nullptr) {
    return absl::InvalidArgumentError("MUSA driver must not be null");
  }
  if (context == nullptr) {
    return absl::InvalidArgumentError("MUSA context must not be null");
  }
  if (context->context() == nullptr) {
    return absl::InvalidArgumentError("native MUSA context must not be null");
  }
  if (architecture.empty()) {
    return absl::InvalidArgumentError(
        "MUSA module architecture must not be empty");
  }
  if (binary_abi_version == 0) {
    return absl::InvalidArgumentError(
        "MUSA module binary ABI version must not be zero");
  }
  return absl::OkStatus();
}

struct MusaModuleCacheKey {
  tsl::Fprint128 content_hash;
  size_t content_size;
  MusaBinaryKind binary_kind;
  std::string architecture;
  MUcontext native_context;
  int device_ordinal;
  MUdevice device;
  uint32_t binary_abi_version;
  MusaMubinMetadata metadata;

  friend bool operator==(const MusaModuleCacheKey& lhs,
                         const MusaModuleCacheKey& rhs) {
    return lhs.content_hash == rhs.content_hash &&
           lhs.content_size == rhs.content_size &&
           lhs.binary_kind == rhs.binary_kind &&
           lhs.architecture == rhs.architecture &&
           lhs.native_context == rhs.native_context &&
           lhs.device_ordinal == rhs.device_ordinal &&
           lhs.device == rhs.device &&
           lhs.binary_abi_version == rhs.binary_abi_version &&
           lhs.metadata.elf_abi_version == rhs.metadata.elf_abi_version &&
           lhs.metadata.elf_machine == rhs.metadata.elf_machine &&
           lhs.metadata.vendor_note_type == rhs.metadata.vendor_note_type;
  }
};

size_t HashCombine(size_t seed, size_t value) {
  // The cache always confirms candidate equality and then compares all bytes;
  // this mixer only selects a bucket and is not an identity boundary.
  return seed ^ (value + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

struct MusaModuleCacheKeyHash {
  size_t operator()(const MusaModuleCacheKey& key) const {
    size_t hash = std::hash<uint64_t>{}(key.content_hash.low64);
    hash = HashCombine(hash, std::hash<uint64_t>{}(key.content_hash.high64));
    hash = HashCombine(hash, std::hash<size_t>{}(key.content_size));
    hash = HashCombine(hash, std::hash<unsigned int>{}(
                                 static_cast<unsigned int>(key.binary_kind)));
    hash = HashCombine(hash, std::hash<std::string>{}(key.architecture));
    hash = HashCombine(hash, std::hash<MUcontext>{}(key.native_context));
    hash = HashCombine(hash, std::hash<int>{}(key.device_ordinal));
    hash = HashCombine(hash, std::hash<MUdevice>{}(key.device));
    hash = HashCombine(hash, std::hash<uint32_t>{}(key.binary_abi_version));
    hash =
        HashCombine(hash, std::hash<uint8_t>{}(key.metadata.elf_abi_version));
    hash = HashCombine(hash, std::hash<uint16_t>{}(key.metadata.elf_machine));
    return HashCombine(hash,
                       std::hash<uint32_t>{}(key.metadata.vendor_note_type));
  }
};

MusaModuleCacheKey MakeCacheKey(absl::Span<const uint8_t> bytes,
                                const MusaMubinMetadata& metadata,
                                const MusaContext& context,
                                absl::string_view architecture,
                                uint32_t binary_abi_version) {
  absl::string_view byte_string(reinterpret_cast<const char*>(bytes.data()),
                                bytes.size());
  return MusaModuleCacheKey{
      /*content_hash=*/tsl::Fingerprint128(byte_string),
      /*content_size=*/bytes.size(),
      /*binary_kind=*/MusaBinaryKind::kMubin,
      /*architecture=*/std::string(architecture),
      /*native_context=*/context.context(),
      /*device_ordinal=*/context.device_ordinal(),
      /*device=*/context.device(),
      /*binary_abi_version=*/binary_abi_version,
      /*metadata=*/metadata,
  };
}

bool HasIdenticalBytes(const MusaModule& module,
                       absl::Span<const uint8_t> bytes) {
  absl::Span<const uint8_t> module_bytes = module.bytes();
  return module_bytes.size() == bytes.size() &&
         std::equal(module_bytes.begin(), module_bytes.end(), bytes.begin());
}

}  // namespace

MusaModule::MusaModule(MusaDriver* driver, std::shared_ptr<MusaContext> context,
                       std::shared_ptr<const std::vector<uint8_t>> bytes,
                       MusaMubinMetadata metadata, std::string architecture,
                       uint32_t binary_abi_version, MUmodule module)
    : driver_(driver),
      context_(std::move(context)),
      bytes_(std::move(bytes)),
      metadata_(metadata),
      architecture_(std::move(architecture)),
      binary_abi_version_(binary_abi_version),
      module_(module) {}

absl::StatusOr<std::shared_ptr<MusaModule>> MusaModule::Load(
    MusaDriver* driver, std::shared_ptr<MusaContext> context,
    absl::Span<const uint8_t> mubin, absl::string_view architecture,
    uint32_t binary_abi_version) {
  RETURN_IF_ERROR(
      ValidateLoadInputs(driver, context, architecture, binary_abi_version));
  TF_ASSIGN_OR_RETURN(MusaMubinMetadata metadata, ValidateMusaMubin(mubin));
  std::shared_ptr<const std::vector<uint8_t>> bytes =
      std::make_shared<const std::vector<uint8_t>>(mubin.begin(), mubin.end());
  return LoadValidated(driver, std::move(context), std::move(bytes), metadata,
                       std::string(architecture), binary_abi_version);
}

absl::StatusOr<std::shared_ptr<MusaModule>> MusaModule::LoadValidated(
    MusaDriver* driver, std::shared_ptr<MusaContext> context,
    std::shared_ptr<const std::vector<uint8_t>> bytes,
    MusaMubinMetadata metadata, std::string architecture,
    uint32_t binary_abi_version) {
  RETURN_IF_ERROR(
      ValidateLoadInputs(driver, context, architecture, binary_abi_version));
  if (bytes == nullptr || bytes->empty()) {
    return absl::InvalidArgumentError(
        "validated MUSA module bytes must not be empty");
  }

  MUmodule native_module;
  {
    gpu::ScopedActivateContext activation(context.get());
    TF_ASSIGN_OR_RETURN(native_module, driver->LoadModuleData(bytes->data()));
  }
  return std::shared_ptr<MusaModule>(new MusaModule(
      driver, std::move(context), std::move(bytes), metadata,
      std::move(architecture), binary_abi_version, native_module));
}

MusaModule::~MusaModule() {
  if (module_ == nullptr) return;
  gpu::ScopedActivateContext activation(context_.get());
  absl::Status status = driver_->UnloadModule(module_);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to unload MUSA module " << module_ << ": " << status;
  }
  module_ = nullptr;
}

struct MusaModuleCache::CacheState {
  struct ClientEntry {
    std::shared_ptr<MusaModule> module;
    size_t client_count;
  };

  struct HandleToken {
    uint64_t serial;
  };

  mutable absl::Mutex mutex;
  absl::flat_hash_map<MusaModuleCacheKey,
                      std::vector<std::weak_ptr<MusaModule>>,
                      MusaModuleCacheKeyHash>
      buckets ABSL_GUARDED_BY(mutex);
  absl::flat_hash_map<const void*, ClientEntry> clients ABSL_GUARDED_BY(mutex);
  // Tokens are never reused during the cache lifetime, so a released handle
  // cannot accidentally alias a subsequently loaded native module.
  std::vector<std::unique_ptr<HandleToken>> tokens ABSL_GUARDED_BY(mutex);
  uint64_t next_token ABSL_GUARDED_BY(mutex) = 1;
  size_t loads_in_flight ABSL_GUARDED_BY(mutex) = 0;

  std::shared_ptr<MusaModule> Find(
      const MusaModuleCacheKey& key, absl::Span<const uint8_t> bytes,
      std::vector<std::shared_ptr<MusaModule>>& keepalive)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex) {
    auto bucket_it = buckets.find(key);
    if (bucket_it == buckets.end()) return nullptr;

    std::vector<std::weak_ptr<MusaModule>>& bucket = bucket_it->second;
    for (auto it = bucket.begin(); it != bucket.end();) {
      std::shared_ptr<MusaModule> module = it->lock();
      if (module == nullptr) {
        it = bucket.erase(it);
        continue;
      }

      // Retain every locked weak reference until after the caller releases
      // the cache mutex. A racing final client release can therefore never
      // make a non-matching module's destructor run under this mutex.
      keepalive.push_back(module);
      if (HasIdenticalBytes(*module, bytes)) return module;
      ++it;
    }
    return nullptr;
  }

  ModuleHandle AddHandleClient(std::shared_ptr<MusaModule> module)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex) {
    for (auto& [token, client] : clients) {
      if (client.module.get() == module.get()) {
        ++client.client_count;
        return ModuleHandle(token);
      }
    }

    auto token = std::make_unique<HandleToken>();
    token->serial = next_token++;
    const void* id = token.get();
    tokens.push_back(std::move(token));
    clients.emplace(id, ClientEntry{std::move(module), 1});
    return ModuleHandle(id);
  }
};

MusaModuleCache::MusaModuleCache(MusaDriver* driver,
                                 std::shared_ptr<MusaContext> context,
                                 std::string architecture,
                                 uint32_t binary_abi_version)
    : driver_(driver),
      context_(std::move(context)),
      architecture_(std::move(architecture)),
      binary_abi_version_(binary_abi_version),
      state_(std::make_unique<CacheState>()) {}

MusaModuleCache::~MusaModuleCache() {
  std::vector<std::shared_ptr<MusaModule>> modules_to_release;
  {
    absl::MutexLock lock(state_->mutex);
    modules_to_release.reserve(state_->clients.size());
    for (auto& [unused, client] : state_->clients) {
      modules_to_release.push_back(std::move(client.module));
    }
    state_->clients.clear();
    state_->buckets.clear();
  }
  // Module destruction activates contexts and calls the driver, so it must
  // happen after releasing the cache mutex.
  modules_to_release.clear();
}

absl::StatusOr<std::shared_ptr<MusaModule>> MusaModuleCache::GetOrLoadModule(
    absl::Span<const uint8_t> mubin) {
  RETURN_IF_ERROR(ValidateLoadInputs(driver_, context_, architecture_,
                                     binary_abi_version_));
  TF_ASSIGN_OR_RETURN(MusaMubinMetadata metadata, ValidateMusaMubin(mubin));
  MusaModuleCacheKey key = MakeCacheKey(mubin, metadata, *context_,
                                        architecture_, binary_abi_version_);

  std::vector<std::shared_ptr<MusaModule>> keepalive;
  std::shared_ptr<MusaModule> selected;
  {
    absl::MutexLock lock(state_->mutex);
    selected = state_->Find(key, mubin, keepalive);
    if (selected == nullptr) ++state_->loads_in_flight;
  }
  if (selected != nullptr) return selected;
  keepalive.clear();

  std::shared_ptr<const std::vector<uint8_t>> owned_bytes =
      std::make_shared<const std::vector<uint8_t>>(mubin.begin(), mubin.end());
  absl::StatusOr<std::shared_ptr<MusaModule>> candidate_or =
      MusaModule::LoadValidated(driver_, context_, owned_bytes, metadata,
                                architecture_, binary_abi_version_);
  if (!candidate_or.ok()) {
    absl::MutexLock lock(state_->mutex);
    --state_->loads_in_flight;
    return candidate_or.status();
  }
  std::shared_ptr<MusaModule> candidate = *std::move(candidate_or);

  {
    absl::MutexLock lock(state_->mutex);
    --state_->loads_in_flight;
    selected = state_->Find(key, mubin, keepalive);
    if (selected == nullptr) {
      state_->buckets[key].push_back(candidate);
      selected = candidate;
    }
  }
  // If another thread won the race, candidate and any non-matching temporary
  // strong references are destroyed here, outside the cache mutex.
  return selected;
}

absl::StatusOr<ModuleHandle> MusaModuleCache::AcquireModuleHandle(
    absl::Span<const uint8_t> mubin) {
  TF_ASSIGN_OR_RETURN(std::shared_ptr<MusaModule> module,
                      GetOrLoadModule(mubin));
  absl::MutexLock lock(state_->mutex);
  return state_->AddHandleClient(std::move(module));
}

absl::StatusOr<std::shared_ptr<MusaModule>> MusaModuleCache::LookupModule(
    ModuleHandle handle) const {
  if (!handle) return absl::NotFoundError("MUSA module handle is null");
  absl::MutexLock lock(state_->mutex);
  auto it = state_->clients.find(handle.id());
  if (it == state_->clients.end()) {
    return absl::NotFoundError("MUSA module handle is not owned by this cache");
  }
  return it->second.module;
}

bool MusaModuleCache::ReleaseModuleHandle(ModuleHandle handle) {
  if (!handle) return false;
  std::shared_ptr<MusaModule> module_to_release;
  {
    absl::MutexLock lock(state_->mutex);
    auto it = state_->clients.find(handle.id());
    if (it == state_->clients.end()) return false;
    if (it->second.client_count > 1) {
      --it->second.client_count;
      return true;
    }
    module_to_release = std::move(it->second.module);
    state_->clients.erase(it);
  }
  // The last shared reference may unload the module, activate its context, and
  // invoke an arbitrary driver implementation. Never perform it under mutex.
  module_to_release.reset();
  return true;
}

bool MusaModuleCache::IsQuiescent() const {
  absl::MutexLock lock(state_->mutex);
  if (!state_->clients.empty() || state_->loads_in_flight != 0) return false;
  for (const auto& [unused, bucket] : state_->buckets) {
    for (const std::weak_ptr<MusaModule>& module : bucket) {
      if (!module.expired()) return false;
    }
  }
  return true;
}

}  // namespace stream_executor::musa
