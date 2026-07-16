// Copyright 2026 The OpenXLA Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xla/stream_executor/module_spec.h"

#include <array>
#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace stream_executor {
namespace {

using ::testing::ElementsAre;

TEST(MultiModuleLoaderSpecTest, AddsNonOwningMusaMubin) {
  std::array<uint8_t, 4> mubin = {0x7f, 'E', 'L', 'F'};
  MultiModuleLoaderSpec spec;

  spec.AddMusaMubinInMemory(mubin);

  EXPECT_TRUE(spec.has_musa_mubin_in_memory());
  EXPECT_FALSE(spec.owns_musa_mubin_in_memory());
  EXPECT_THAT(spec.musa_mubin_in_memory(), ElementsAre(0x7f, 'E', 'L', 'F'));

  mubin[1] = 'e';
  EXPECT_THAT(spec.musa_mubin_in_memory(), ElementsAre(0x7f, 'e', 'L', 'F'));
}

TEST(MultiModuleLoaderSpecTest, OwnsMusaMubinAcrossCopies) {
  MultiModuleLoaderSpec original;
  original.AddOwningMusaMubinInMemory(
      std::vector<uint8_t>{0x7f, 'E', 0x00, 'F'});

  MultiModuleLoaderSpec copy = original;
  original.AddOwningMusaMubinInMemory(std::vector<uint8_t>{1, 2, 3});

  EXPECT_TRUE(copy.has_musa_mubin_in_memory());
  EXPECT_TRUE(copy.owns_musa_mubin_in_memory());
  EXPECT_THAT(copy.musa_mubin_in_memory(), ElementsAre(0x7f, 'E', 0x00, 'F'));
}

TEST(MultiModuleLoaderSpecTest, MusaAndCudaArtifactsRemainDistinct) {
  std::array<uint8_t, 2> cubin = {1, 2};
  MultiModuleLoaderSpec spec;
  spec.AddCudaCubinInMemory(cubin);

  EXPECT_TRUE(spec.has_cuda_cubin_in_memory());
  EXPECT_FALSE(spec.has_musa_mubin_in_memory());
}

}  // namespace
}  // namespace stream_executor
