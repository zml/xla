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

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/stream_executor/semantic_version.h"

namespace stream_executor::musa {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

TEST(MusaVersionParserTest, ParsesRuntimeAndSdkEncodings) {
  EXPECT_THAT(ParseMusaVersion(10504), IsOkAndHolds(SemanticVersion{1, 5, 4}));
  EXPECT_THAT(ParseMusaVersion(30000), IsOkAndHolds(SemanticVersion{3, 0, 0}));
  EXPECT_THAT(ParseMusaVersion(40001), IsOkAndHolds(SemanticVersion{4, 0, 1}));
  EXPECT_THAT(ParseMusaVersion(0), IsOkAndHolds(SemanticVersion{0, 0, 0}));
}

TEST(MusaVersionParserTest, RejectsNegativeVersion) {
  EXPECT_THAT(ParseMusaVersion(-1),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(MusaVersionParserTest, ParsesKernelDriverProcFile) {
  EXPECT_THAT(ParseMusaKernelDriverVersion("package version:3.0.0\n"),
              IsOkAndHolds(SemanticVersion{3, 0, 0}));
  EXPECT_THAT(ParseMusaKernelDriverVersion("package version: 3.0.0\n"),
              IsOkAndHolds(SemanticVersion{3, 0, 0}));
}

TEST(MusaVersionParserTest, RejectsMalformedKernelDriverVersion) {
  EXPECT_THAT(ParseMusaKernelDriverVersion("build version:unknown\n"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace stream_executor::musa
