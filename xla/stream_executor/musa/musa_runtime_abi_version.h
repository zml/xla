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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/abi/executable_abi_version.h"
#include "xla/stream_executor/abi/runtime_abi_version.h"
#include "xla/stream_executor/abi/runtime_abi_version.pb.h"
#include "xla/stream_executor/platform_id.h"
#include "xla/stream_executor/semantic_version.h"

namespace stream_executor::musa {

class MusaRuntimeAbiVersion : public RuntimeAbiVersion {
 public:
  explicit MusaRuntimeAbiVersion(SemanticVersion runtime_version)
      : runtime_version_(runtime_version) {}

  absl::Status IsCompatibleWith(
      const ExecutableAbiVersion& executable_abi_version) const override;

  absl::StatusOr<RuntimeAbiVersionProto> ToProto() const override;
  absl::StatusOr<PlatformId> platform_id() const override;

 private:
  SemanticVersion runtime_version_;
};

}  // namespace stream_executor::musa

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_RUNTIME_ABI_VERSION_H_
