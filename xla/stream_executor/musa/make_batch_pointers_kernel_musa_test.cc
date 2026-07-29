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

#include <cstdint>

#include <gtest/gtest.h>
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/gpu/make_batch_pointers_kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/musa/musa_platform_id.h"

namespace stream_executor::musa {
namespace {

TEST(MakeBatchPointersKernelMusaTest, RegistersExplicitMubin) {
  auto found = gpu::GpuKernelRegistry::GetGlobalRegistry()
                   .FindKernel<gpu::MakeBatchPointersKernel>(kMusaPlatformId);
  ASSERT_TRUE(found.ok()) << found.status();

  const KernelLoaderSpec& spec = found->get();
  EXPECT_EQ(spec.arity(), 4);
  EXPECT_EQ(spec.kernel_name(), "make_batch_pointers");
  EXPECT_TRUE(spec.has_musa_mubin_in_memory());
  EXPECT_FALSE(spec.has_in_process_symbol());

  ASSERT_TRUE(spec.musa_mubin_in_memory().has_value());
  const absl::Span<const uint8_t> bytes =
      spec.musa_mubin_in_memory()->mubin_bytes;
  ASSERT_GE(bytes.size(), 4);
  EXPECT_EQ(bytes[0], 0x7f);
  EXPECT_EQ(bytes[1], 'E');
  EXPECT_EQ(bytes[2], 'L');
  EXPECT_EQ(bytes[3], 'F');
}

}  // namespace
}  // namespace stream_executor::musa
