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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_DEVICE_DESCRIPTION_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_DEVICE_DESCRIPTION_H_

#include <optional>
#include <string>

#include "absl/status/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/musa/musa_device_properties.h"
#include "xla/stream_executor/semantic_version.h"

namespace stream_executor::musa {

struct MusaDeviceVersions {
  int runtime_api = 0;
  int driver_api = 0;
  int compile_time_toolkit = 0;
  std::optional<SemanticVersion> kernel_mode_driver;
};

absl::StatusOr<std::string> MusaArchitectureFromComputeCapability(int major,
                                                                  int minor);

// Converts raw queried device facts to XLA's shared DeviceDescription. Keeping
// this conversion pure makes the target contract host-testable without a MUSA
// driver or device.
absl::StatusOr<DeviceDescription> BuildMusaDeviceDescription(
    const MusaDeviceProperties& properties, const MusaDeviceVersions& versions);

}  // namespace stream_executor::musa

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_DEVICE_DESCRIPTION_H_
