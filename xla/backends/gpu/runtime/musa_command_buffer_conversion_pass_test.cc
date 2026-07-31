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
#include <memory>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/command_buffer_conversion_pass.h"
#include "xla/backends/gpu/runtime/device_to_device_copy_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_pass_pipeline.h"
#include "xla/debug_options_flags.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/musa/musa_compute_capability.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

class FailingAllocator : public ThunkPassBufferAllocator {
 public:
  absl::StatusOr<BufferAllocation*> NewEmptyAllocation(int64_t size) override {
    return absl::InternalError("allocation is not expected");
  }
};

TEST(MusaCommandBufferConversionPassTest, ConvertsQualifiedFusionCommands) {
  BufferAllocation allocation(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice(&allocation, /*offset=*/0, /*size=*/1024);
  Shape shape = ShapeUtil::MakeShape(S32, {256});

  ThunkSequence thunks;
  thunks.push_back(std::make_unique<DeviceToDeviceCopyThunk>(
      Thunk::ThunkInfo(), ShapedSlice{slice, shape}, ShapedSlice{slice, shape},
      /*mem_size=*/1024));

  DebugOptions debug_options = GetDebugOptionsFromFlags();
  debug_options.set_xla_gpu_graph_min_graph_size(1);
  debug_options.clear_xla_gpu_enable_command_buffer();
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);

  se::DeviceDescription device_info;
  device_info.set_gpu_compute_capability(se::GpuComputeCapability(
      se::MusaComputeCapability("mp_21", /*major=*/2, /*minor=*/1,
                                /*hardware_warp_size=*/128,
                                /*logical_subgroup_size=*/32)));

  FailingAllocator allocator;
  CommandBufferConversionPass pass{"test"};
  absl::StatusOr<bool> changed = pass.Run(
      &thunks, debug_options, /*hlo_module=*/nullptr, device_info, allocator);

  ASSERT_TRUE(changed.ok()) << changed.status();
  EXPECT_TRUE(*changed);
  ASSERT_EQ(thunks.size(), 1);
  EXPECT_EQ(thunks.front()->kind(), Thunk::kCommandBuffer);
}

TEST(MusaCommandBufferConversionPassTest,
     LeavesUnqualifiedLibraryCommandsOnStreams) {
  BufferAllocation allocation(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice(&allocation, /*offset=*/0, /*size=*/1024);
  Shape shape = ShapeUtil::MakeShape(S32, {256});

  ThunkSequence thunks;
  thunks.push_back(std::make_unique<DeviceToDeviceCopyThunk>(
      Thunk::ThunkInfo(), ShapedSlice{slice, shape}, ShapedSlice{slice, shape},
      /*mem_size=*/1024));

  DebugOptions debug_options = GetDebugOptionsFromFlags();
  debug_options.set_xla_gpu_graph_min_graph_size(1);
  debug_options.clear_xla_gpu_enable_command_buffer();
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::CUBLAS);

  se::DeviceDescription device_info;
  device_info.set_gpu_compute_capability(se::GpuComputeCapability(
      se::MusaComputeCapability("mp_21", /*major=*/2, /*minor=*/1,
                                /*hardware_warp_size=*/128,
                                /*logical_subgroup_size=*/32)));

  FailingAllocator allocator;
  CommandBufferConversionPass pass{"test"};
  absl::StatusOr<bool> changed = pass.Run(
      &thunks, debug_options, /*hlo_module=*/nullptr, device_info, allocator);

  ASSERT_TRUE(changed.ok()) << changed.status();
  EXPECT_FALSE(*changed);
  ASSERT_EQ(thunks.size(), 1);
  EXPECT_EQ(thunks.front()->kind(), Thunk::kCopy);
}

TEST(MusaCommandBufferConversionPassTest,
     DoesNotAggregateIndependentRuntimeThunks) {
  BufferAllocation allocation(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice slice(&allocation, /*offset=*/0, /*size=*/1024);
  Shape shape = ShapeUtil::MakeShape(S32, {256});

  ThunkSequence thunks;
  thunks.push_back(std::make_unique<DeviceToDeviceCopyThunk>(
      Thunk::ThunkInfo(), ShapedSlice{slice, shape}, ShapedSlice{slice, shape},
      /*mem_size=*/1024));
  thunks.push_back(std::make_unique<DeviceToDeviceCopyThunk>(
      Thunk::ThunkInfo(), ShapedSlice{slice, shape}, ShapedSlice{slice, shape},
      /*mem_size=*/1024));

  DebugOptions debug_options = GetDebugOptionsFromFlags();
  debug_options.set_xla_gpu_graph_min_graph_size(1);
  debug_options.clear_xla_gpu_enable_command_buffer();
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);

  se::DeviceDescription device_info;
  device_info.set_gpu_compute_capability(se::GpuComputeCapability(
      se::MusaComputeCapability("mp_21", /*major=*/2, /*minor=*/1,
                                /*hardware_warp_size=*/128,
                                /*logical_subgroup_size=*/32)));

  FailingAllocator allocator;
  CommandBufferConversionPass pass{"test"};
  absl::StatusOr<bool> changed = pass.Run(
      &thunks, debug_options, /*hlo_module=*/nullptr, device_info, allocator);

  ASSERT_TRUE(changed.ok()) << changed.status();
  EXPECT_TRUE(*changed);
  ASSERT_EQ(thunks.size(), 2);
  EXPECT_EQ(thunks[0]->kind(), Thunk::kCommandBuffer);
  EXPECT_EQ(thunks[1]->kind(), Thunk::kCommandBuffer);
}

}  // namespace
}  // namespace xla::gpu
