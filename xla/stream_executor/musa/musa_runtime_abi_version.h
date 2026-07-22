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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_RUNTIME_ABI_VERSION_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_RUNTIME_ABI_VERSION_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/abi/executable_abi_version.h"
#include "xla/stream_executor/abi/runtime_abi_version.h"
#include "xla/stream_executor/abi/runtime_abi_version.pb.h"
#include "xla/stream_executor/musa/musa_runtime_abi_version.pb.h"
#include "xla/stream_executor/platform_id.h"
#include "xla/stream_executor/semantic_version.h"

namespace stream_executor::musa {

struct MusaOptionalLibraryAbi {
  std::string name;
  std::string abi_version;
  std::string fingerprint;

  bool operator==(const MusaOptionalLibraryAbi& other) const {
    return name == other.name && abi_version == other.abi_version &&
           fingerprint == other.fingerprint;
  }
};

class MusaRuntimeAbiVersion : public RuntimeAbiVersion {
 public:
  // Creates an ABI snapshot from the version values returned by the MUSA
  // runtime/driver APIs and the compile-time SDK encoding. No device or context
  // is needed to construct this snapshot.
  static absl::StatusOr<MusaRuntimeAbiVersion> CreateFromApiVersions(
      int runtime_version, int driver_version,
      SemanticVersion kernel_driver_version, int toolkit_version);

  static absl::StatusOr<MusaRuntimeAbiVersion> Create(
      SemanticVersion runtime_version, SemanticVersion driver_version,
      SemanticVersion kernel_driver_version, SemanticVersion toolkit_version,
      std::vector<MusaOptionalLibraryAbi> available_optional_library_abis = {});

  MusaRuntimeAbiVersion(const MusaRuntimeAbiVersion&) = default;
  MusaRuntimeAbiVersion(MusaRuntimeAbiVersion&&) = default;
  MusaRuntimeAbiVersion& operator=(const MusaRuntimeAbiVersion&) = default;
  MusaRuntimeAbiVersion& operator=(MusaRuntimeAbiVersion&&) = default;

  static absl::StatusOr<std::unique_ptr<MusaRuntimeAbiVersion> absl_nonnull>
  FromProto(const MusaRuntimeAbiVersionProto& proto);
  static absl::StatusOr<std::unique_ptr<MusaRuntimeAbiVersion> absl_nonnull>
  FromSerializedProto(absl::string_view proto);

  absl::Status IsCompatibleWith(
      const ExecutableAbiVersion& executable_abi_version) const override;

  absl::StatusOr<RuntimeAbiVersionProto> ToProto() const override;
  absl::StatusOr<PlatformId> platform_id() const override;

  const SemanticVersion& runtime_version() const { return runtime_version_; }
  const SemanticVersion& driver_version() const { return driver_version_; }
  const SemanticVersion& kernel_driver_version() const {
    return kernel_driver_version_;
  }
  const SemanticVersion& toolkit_version() const { return toolkit_version_; }
  const std::vector<MusaOptionalLibraryAbi>& available_optional_library_abis()
      const {
    return available_optional_library_abis_;
  }

 private:
  struct ValidatedTag {};
  MusaRuntimeAbiVersion(
      ValidatedTag, SemanticVersion runtime_version,
      SemanticVersion driver_version, SemanticVersion kernel_driver_version,
      SemanticVersion toolkit_version,
      std::vector<MusaOptionalLibraryAbi> available_optional_library_abis);

  SemanticVersion runtime_version_;
  SemanticVersion driver_version_;
  SemanticVersion kernel_driver_version_;
  SemanticVersion toolkit_version_;
  std::vector<MusaOptionalLibraryAbi> available_optional_library_abis_;
};

}  // namespace stream_executor::musa

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_RUNTIME_ABI_VERSION_H_
