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

#include "xla/stream_executor/musa/musa_optional_libraries.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/stream_executor/musa/musa_optional_library_abi.h"

namespace stream_executor::musa {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::HasSubstr;

TEST(MusaOptionalLibrariesTest, MergesInCanonicalNameOrder) {
  const std::vector<std::vector<MusaOptionalLibraryAbi>> providers = {
      {{"mufft", "1", std::string(64, 'f')}},
      {{"mublas", "1", ""},
       {"mublas-scal", "1", std::string(64, 's')},
       {"mublas-trsm", "1", std::string(64, 't')}}};
  ASSERT_OK_AND_ASSIGN(std::vector<MusaOptionalLibraryAbi> merged,
                       internal::MergeMusaOptionalLibraryAbis(providers));
  EXPECT_THAT(merged,
              ElementsAre(Field(&MusaOptionalLibraryAbi::name, "mublas"),
                          Field(&MusaOptionalLibraryAbi::name, "mublas-scal"),
                          Field(&MusaOptionalLibraryAbi::name, "mublas-trsm"),
                          Field(&MusaOptionalLibraryAbi::name, "mufft")));
}

TEST(MusaOptionalLibrariesTest, RejectsDuplicateProviderNames) {
  const std::vector<std::vector<MusaOptionalLibraryAbi>> providers = {
      {{"mufft", "1", std::string(64, 'a')}},
      {{"mufft", "2", std::string(64, 'b')}}};
  EXPECT_THAT(
      internal::MergeMusaOptionalLibraryAbis(providers),
      StatusIs(absl::StatusCode::kFailedPrecondition, HasSubstr("duplicate")));
}

TEST(MusaOptionalLibrariesTest, RejectsEmptyIdentityFields) {
  const std::vector<std::vector<MusaOptionalLibraryAbi>> providers = {
      {{"", "1", ""}}};
  EXPECT_THAT(
      internal::MergeMusaOptionalLibraryAbis(providers),
      StatusIs(absl::StatusCode::kFailedPrecondition, HasSubstr("empty")));
}

}  // namespace
}  // namespace stream_executor::musa
