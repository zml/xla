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

#include "xla/stream_executor/musa/musa_runtime_abi_version.h"

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/stream_executor/abi/executable_abi_version.h"
#include "xla/stream_executor/abi/executable_abi_version.pb.h"
#include "xla/stream_executor/abi/runtime_abi_version.pb.h"
#include "xla/stream_executor/musa/musa_platform_id.h"
#include "xla/stream_executor/platform_id.h"

namespace stream_executor::musa {

absl::Status MusaRuntimeAbiVersion::IsCompatibleWith(
    const ExecutableAbiVersion& executable_abi_version) const {
  const ExecutableAbiVersionProto& executable_proto =
      executable_abi_version.proto();
  if (executable_proto.platform_name() != kMusaPlatformId->ToName()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Platform name mismatch. Expected ", kMusaPlatformId->ToName(),
        ", but got ", executable_proto.platform_name()));
  }
  return absl::OkStatus();
}

absl::StatusOr<RuntimeAbiVersionProto> MusaRuntimeAbiVersion::ToProto() const {
  RuntimeAbiVersionProto proto;
  proto.set_platform_name(kMusaPlatformId->ToName());
  proto.set_platform_specific_version(runtime_version_.ToString());
  return proto;
}

absl::StatusOr<PlatformId> MusaRuntimeAbiVersion::platform_id() const {
  return kMusaPlatformId;
}

}  // namespace stream_executor::musa
