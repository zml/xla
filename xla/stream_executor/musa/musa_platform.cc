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

#include "xla/stream_executor/musa/musa_platform.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/abi/runtime_abi_version.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/musa/musa_executor.h"
#include "xla/stream_executor/musa/musa_platform_id.h"
#include "xla/stream_executor/musa/musa_runtime.h"
#include "xla/stream_executor/musa/musa_runtime_abi_version.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/status_macros.h"

namespace stream_executor::musa {
namespace {

SemanticVersion VersionFromRuntimeInt(int version) {
  if (version <= 0) {
    return SemanticVersion{0, 0, 0};
  }
  return SemanticVersion{static_cast<unsigned>(version / 1000),
                         static_cast<unsigned>((version % 1000) / 10),
                         static_cast<unsigned>(version % 10)};
}

}  // namespace

MusaPlatform::MusaPlatform() : name_(kMusaPlatformId->ToName()) {}

Platform::Id MusaPlatform::id() const { return kMusaPlatformId; }

int MusaPlatform::VisibleDeviceCount() const {
  auto count = MusaRuntime::Get()->GetDeviceCount();
  if (!count.ok()) {
    LOG(ERROR) << "Failed to get MUSA device count: " << count.status();
    return -1;
  }
  return *count;
}

const std::string& MusaPlatform::Name() const { return name_; }

absl::StatusOr<std::unique_ptr<DeviceDescription>>
MusaPlatform::DescriptionForDevice(int ordinal) const {
  return MusaExecutor::CreateDeviceDescription(ordinal);
}

absl::StatusOr<StreamExecutor*> MusaPlatform::ExecutorForDevice(int ordinal) {
  return executor_cache_.GetOrCreate(
      ordinal, [this, ordinal]() { return GetUncachedExecutor(ordinal); });
}

absl::StatusOr<StreamExecutor*> MusaPlatform::FindExisting(int ordinal) {
  return executor_cache_.Get(ordinal);
}

absl::StatusOr<std::unique_ptr<RuntimeAbiVersion>>
MusaPlatform::GetRuntimeAbiVersion() const {
  SemanticVersion runtime_version{0, 0, 0};
  if (auto version = MusaRuntime::Get()->RuntimeVersion(); version.ok()) {
    runtime_version = VersionFromRuntimeInt(*version);
  }
  return std::make_unique<MusaRuntimeAbiVersion>(runtime_version);
}

absl::StatusOr<std::unique_ptr<StreamExecutor>>
MusaPlatform::GetUncachedExecutor(int ordinal) {
  auto executor = std::make_unique<MusaExecutor>(this, ordinal);
  TF_RETURN_IF_ERROR(executor->Init());
  return std::move(executor);
}

}  // namespace stream_executor::musa

namespace stream_executor {

static void InitializeMusaPlatform() {
  auto status = PlatformManager::PlatformWithName("MUSA");
  if (!status.ok()) {
    CHECK_OK(PlatformManager::RegisterPlatform(
        std::make_unique<musa::MusaPlatform>()));
  }
}

}  // namespace stream_executor

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(
    musa_platform, stream_executor::InitializeMusaPlatform());
