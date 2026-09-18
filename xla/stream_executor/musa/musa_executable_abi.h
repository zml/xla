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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_EXECUTABLE_ABI_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_EXECUTABLE_ABI_H_

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/stream_executor/abi/executable_abi_version.h"

namespace stream_executor::musa {

// Returns the canonical lower-case SHA-256 digest used to authenticate the
// main device binary in a MUSA executable envelope.
std::string MusaExecutableBinarySha256(absl::Span<const uint8_t> main_binary);

// Validates the loader-visible, platform-independent part of a MUSA
// executable envelope and authenticates the main MUBIN. Compiler provenance
// is checked for a bounded, serializable shape, but is deliberately treated as
// opaque: loading an executable must not require the loader to have the same
// compiler bridge, shim, mapping, or LLVM compatibility implementation that
// produced it.
absl::Status ValidateMusaExecutableAbi(
    const ExecutableAbiVersion& executable_abi_version,
    absl::Span<const uint8_t> main_binary);

}  // namespace stream_executor::musa

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_EXECUTABLE_ABI_H_
