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

#include "xla/backends/gpu/codegen/kernels/musa_custom_kernel.h"

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu::kernel {
namespace {

namespace se = ::stream_executor;
using ::testing::ElementsAre;

TEST(MusaCustomKernelTest, PreservesOwningMubinArtifactType) {
  TF_ASSERT_OK_AND_ASSIGN(
      CustomKernel custom_kernel,
      CreateOwnedMubinCustomKernel("add", {0x7f, 'E', 'L', 'F'}, 2,
                                   se::BlockDim(1), se::ThreadDim(32), 0));

  const se::KernelLoaderSpec& spec = custom_kernel.kernel_spec();
  EXPECT_TRUE(spec.has_musa_mubin_in_memory());
  EXPECT_FALSE(spec.has_cuda_cubin_in_memory());
  ASSERT_TRUE(spec.musa_mubin_in_memory().has_value());
  EXPECT_THAT(spec.musa_mubin_in_memory()->mubin_bytes,
              ElementsAre(0x7f, 'E', 'L', 'F'));
  EXPECT_EQ(spec.arity(), 2);
}
