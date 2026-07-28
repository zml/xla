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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_OPTIONAL_LIBRARIES_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_OPTIONAL_LIBRARIES_H_

#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/stream_executor/musa/musa_optional_library_abi.h"

namespace stream_executor::musa {

namespace internal {

// Merges independently discovered provider records into the deterministic
// runtime ABI order. Duplicate names are rejected because an executable names
// exactly one compatibility contract for each optional library.
absl::StatusOr<std::vector<MusaOptionalLibraryAbi>>
MergeMusaOptionalLibraryAbis(
    absl::Span<const std::vector<MusaOptionalLibraryAbi>> providers);

}  // namespace internal

// Discovers every optional MUSA vendor-library adapter without making any one
// of them a plugin-load dependency. Normal absence produces no record;
// explicitly configured or present-but-malformed adapters fail closed.
absl::StatusOr<std::vector<MusaOptionalLibraryAbi>>
GetAvailableMusaOptionalLibraryAbis();

}  // namespace stream_executor::musa

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_OPTIONAL_LIBRARIES_H_
