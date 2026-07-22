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

#ifndef XLA_SERVICE_GPU_MUSA_MUSA_EXECUTABLE_ENVELOPE_H_
#define XLA_SERVICE_GPU_MUSA_MUSA_EXECUTABLE_ENVELOPE_H_

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/service/gpu/musa/musa_compilation_provider.h"
#include "xla/stream_executor/abi/executable_abi_version.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/musa/musa_target_contract.h"

namespace xla::gpu::musa {

inline constexpr uint32_t kMusaExecutableEnvelopeVersion =
    stream_executor::musa::kMusaExecutableAbiEnvelopeVersion;

struct MusaOptionalLibraryAbi {
  std::string name;
  std::string abi_version;
  std::string fingerprint;
};

// Builds the compiler-produced, runtime-independent MUSA executable identity.
// Compiler and bridge fingerprints are provenance and cache identity; they are
// intentionally not required to be installed when the executable is loaded.
absl::StatusOr<stream_executor::ExecutableAbiVersion>
BuildMusaExecutableEnvelope(
    const stream_executor::DeviceDescription& device_description,
    const MusaCompilationIdentity& identity,
    const MusaCompilationCapabilities& capabilities,
    const MusaCompilationOptions& compilation_options,
    absl::Span<const uint8_t> main_binary,
    absl::Span<const MusaOptionalLibraryAbi> required_optional_libraries = {});

// Validates the loader-visible ABI and the exact compiler production
// constants used by this XLA revision. Runtime version/library compatibility
// is validated separately by MusaRuntimeAbiVersion. Loaders must call the
// lightweight stream_executor::musa::ValidateMusaExecutableAbi instead.
absl::Status ValidateMusaExecutableEnvelope(
    const stream_executor::ExecutableAbiVersion& executable_abi_version,
    absl::Span<const uint8_t> main_binary);

}  // namespace xla::gpu::musa

#endif  // XLA_SERVICE_GPU_MUSA_MUSA_EXECUTABLE_ENVELOPE_H_
