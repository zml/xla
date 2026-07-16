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

#ifndef XLA_STREAM_EXECUTOR_MUSA_MUSA_VERSION_PARSER_H_
#define XLA_STREAM_EXECUTOR_MUSA_MUSA_VERSION_PARSER_H_

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/semantic_version.h"

namespace stream_executor::musa {

// Parses the integer returned by MUSA runtime and driver version APIs. MUSA
// encodes versions as major * 10000 + minor * 100 + patch.
absl::StatusOr<SemanticVersion> ParseMusaVersion(int musa_version);

// Parses the Linux kernel driver's /proc/driver/musa/version contents, whose
// stable public line is "package version:X.Y.Z".
absl::StatusOr<SemanticVersion> ParseMusaKernelDriverVersion(
    absl::string_view contents);

}  // namespace stream_executor::musa

#endif  // XLA_STREAM_EXECUTOR_MUSA_MUSA_VERSION_PARSER_H_
