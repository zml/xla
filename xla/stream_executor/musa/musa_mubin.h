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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_MUBIN_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_MUBIN_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace stream_executor::musa {

// Version of XLA's checked MUBIN loader contract. This is intentionally
// independent of the ELF ABI version stored in a MUBIN's ELF identification.
inline constexpr uint32_t kMubinLoaderAbiVersion = 1;

// Target metadata authenticated by ValidateMusaMubin before the bytes are
// handed to the MUSA driver.
struct MusaMubinMetadata {
  uint8_t elf_abi_version;
  uint16_t elf_machine;
  uint32_t vendor_note_type;
};

// Validates the platform-independent envelope facts needed to identify a
// MUBIN. The vendor note payload is deliberately left opaque: interpreting it
// is the vendor loader's responsibility.
absl::StatusOr<MusaMubinMetadata> ValidateMusaMubin(
    absl::Span<const uint8_t> bytes);

}  // namespace stream_executor::musa

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_MUBIN_H_
