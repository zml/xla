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

#include "xla/service/gpu/fly_call.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using ::testing::ElementsAre;

TEST(FlyCallTest, ParsesLaunchAndOccupancyMetadata) {
  mlir::MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(
      FlyCall call,
      FlyCall::Parse(
          R"mlir({name = "kernel", ir = "module {}", num_warps = 4 : i32,
                  grid_x = 8 : i32, grid_y = 2 : i32, grid_z = 1 : i32,
                  waves_per_eu = 2 : i32,
                  shared_mem_bytes = 4096 : i64,
                  zeroed_outputs = [0 : i32, 2 : i32]})mlir",
          &context));

  EXPECT_EQ(call.name, "kernel");
  EXPECT_EQ(call.ir, "module {}");
  EXPECT_EQ(call.num_warps, 4);
  EXPECT_EQ(call.grid_x, 8);
  EXPECT_EQ(call.grid_y, 2);
  EXPECT_EQ(call.grid_z, 1);
  EXPECT_EQ(call.waves_per_eu, 2);
  EXPECT_EQ(call.shared_mem_bytes, 4096);
  EXPECT_THAT(call.zeroed_outputs, ElementsAre(0, 2));
}

TEST(FlyCallTest, RejectsMissingLaunchMetadata) {
  mlir::MLIRContext context;
  auto call = FlyCall::Parse(R"mlir({name = "kernel", ir = "module {}"})mlir",
                             &context);
  EXPECT_FALSE(call.ok());
}

}  // namespace
}  // namespace xla::gpu
