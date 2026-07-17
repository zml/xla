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

#include "xla/codegen/device_spec.h"

#include <gtest/gtest.h>
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/musa/musa_compute_capability.h"

namespace xla {
namespace {

TEST(DeviceSpecTest, IdentifiesMusaGpuWithoutTargetFallback) {
  stream_executor::DeviceDescription device;
  device.set_gpu_compute_capability(stream_executor::GpuComputeCapability(
      stream_executor::MusaComputeCapability("mp_21", 2, 1,
                                             /*hardware_warp_size=*/128,
                                             /*logical_subgroup_size=*/32)));

  DeviceSpec spec(device);
  EXPECT_TRUE(spec.IsGpu());
  EXPECT_TRUE(spec.IsMusaGpu());
  EXPECT_FALSE(spec.IsNvidiaGpu());
  EXPECT_FALSE(spec.IsAmdGpu());
  EXPECT_FALSE(spec.IsIntelGpu());
  EXPECT_FALSE(spec.IsCpu());
}

TEST(DeviceSpecTest, CpuIsNotClassifiedAsMusa) {
  DeviceSpec spec(CpuDeviceSpec{});
  EXPECT_TRUE(spec.IsCpu());
  EXPECT_FALSE(spec.IsGpu());
  EXPECT_FALSE(spec.IsMusaGpu());
}

}  // namespace
}  // namespace xla
