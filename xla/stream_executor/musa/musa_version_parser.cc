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

#include "xla/stream_executor/musa/musa_version_parser.h"

#include <array>
#include <cstddef>
#include <cstdio>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/semantic_version.h"

namespace stream_executor::musa {

absl::StatusOr<SemanticVersion> ParseMusaVersion(int musa_version) {
  if (musa_version < 0) {
    return absl::InvalidArgumentError(
        "MUSA version numbers cannot be negative");
  }
  return SemanticVersion{static_cast<unsigned>(musa_version / 10000),
                         static_cast<unsigned>((musa_version % 10000) / 100),
                         static_cast<unsigned>(musa_version % 100)};
}

absl::StatusOr<SemanticVersion> ParseMusaKernelDriverVersion(
    absl::string_view contents) {
  constexpr absl::string_view kPrefix = "package version:";
  size_t begin = contents.find(kPrefix);
  if (begin == absl::string_view::npos) {
    return absl::InvalidArgumentError(
        "MUSA kernel driver version does not contain 'package version:'");
  }
  begin += kPrefix.size();
  absl::string_view version =
      absl::StripLeadingAsciiWhitespace(contents.substr(begin));
  size_t end = version.find_first_of("\r\n \t");
  version = version.substr(0, end);
  if (version.empty()) {
    return absl::InvalidArgumentError("MUSA kernel driver version is empty");
  }
  return SemanticVersion::ParseFromString(version);
}

absl::StatusOr<SemanticVersion> GetMusaKernelDriverVersion() {
  std::FILE* file = std::fopen("/proc/driver/musa/version", "r");
  if (file == nullptr) {
    return absl::UnavailableError("Could not open /proc/driver/musa/version");
  }
  std::array<char, 256> contents = {};
  const char* read_result = std::fgets(contents.data(), contents.size(), file);
  std::fclose(file);
  if (read_result == nullptr) {
    return absl::DataLossError("Could not read /proc/driver/musa/version");
  }
  return ParseMusaKernelDriverVersion(contents.data());
}

}  // namespace stream_executor::musa
