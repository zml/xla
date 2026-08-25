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
#include <vector>

#include <gtest/gtest.h>
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_kernel_thunk.h"
#include "xla/backends/gpu/runtime/collective_memory_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_init.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

TEST(CollectiveKernelThunkPrepareTest, PeerKernelSkipsCliqueRequest) {
  TF_ASSERT_OK_AND_ASSIGN(
      se::Platform * platform,
      se::PlatformManager::PlatformWithName(se::GpuPlatformName()));
  TF_ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor,
                          platform->ExecutorForDevice(0));
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  DeviceAssignment device_assignment(/*replica_count=*/1,
                                     /*computation_count=*/1);
  device_assignment(0, 0) = 0;
  GpuExecutableRunOptions gpu_options;
  gpu_options.set_gpu_global_device_ids(GpuExecutableRunOptions::DeviceIdMap{
      {LocalDeviceId(0), GlobalDeviceId(0)}});
  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  run_options.mutable_run_options()->set_device_assignment(&device_assignment);
  run_options.mutable_run_options()->set_gpu_executable_run_options(
      &gpu_options);
  run_options.mutable_run_options()->set_local_device_count(1);
  TF_ASSERT_OK_AND_ASSIGN(
      CollectiveParams collective_params,
      CollectiveParams::Create(run_options, /*async_streams=*/{},
                               LocalDeviceId(executor->device_ordinal())));

  constexpr int64_t kElements = 64;
  constexpr int64_t kBytes = kElements * sizeof(float);
  BufferAllocation allocation(/*index=*/0, /*size=*/2 * kBytes, /*color=*/0);
  Shape shape = ShapeUtil::MakeShape(F32, {kElements});
  std::vector<CollectiveThunk::Buffer> buffers = {{
      /*element_count=*/kElements,
      /*source_buffer=*/
      {BufferAllocation::Slice(&allocation, /*offset=*/0, kBytes), shape},
      /*destination_buffer=*/
      {BufferAllocation::Slice(&allocation, /*offset=*/kBytes, kBytes), shape},
      /*source_memory_space=*/0,
      /*destination_memory_space=*/0,
  }};

  ReplicaGroup replica_group;
  replica_group.add_replica_ids(0);
  CollectiveConfig collective_config{
      /*operand_element_type=*/{F32},
      /*replica_groups=*/{replica_group},
      /*group_mode=*/
      CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA,
      /*use_symmetric_buffer=*/false};
  CollectiveKernelSpec kernel_spec{
      /*input_buffer_specs=*/{{false, SymmetricMemoryType::kNone}},
      /*output_buffer_specs=*/{{false, SymmetricMemoryType::kNone}},
      /*scratch_buffers=*/{},
      /*argument_descriptors=*/{},
  };

  auto make_thunk = [&](bool skip_collective_clique) {
    return std::make_unique<CollectiveKernelThunk>(
        Thunk::ThunkInfo(), collective_config, kernel_spec,
        /*is_async=*/false, buffers,
        /*is_collective_kernel_enabled=*/true,
        /*kernel_name=*/"peer_kernel", LaunchDimensions(1, 1),
        /*shmem_bytes=*/0, /*cubin=*/std::nullopt,
        /*use_pdl=*/false, skip_collective_clique);
  };

  se::DeviceAddressBase device_buffer =
      executor->AllocateArray<uint8_t>(2 * kBytes);
  ASSERT_FALSE(device_buffer.is_null());
  BufferAllocations buffer_allocations({device_buffer},
                                       executor->device_ordinal(), nullptr);

  CollectiveCliqueRequests regular_clique_requests;
  CollectiveMemoryRequests regular_memory_requests(buffer_allocations);
  Thunk::PrepareParams regular_params{
      &collective_params, &regular_clique_requests, &regular_memory_requests,
      executor, &buffer_allocations};
  auto regular_status =
      make_thunk(/*skip_collective_clique=*/false)->Prepare(regular_params);
  ASSERT_TRUE(regular_status.ok()) << regular_status;
  EXPECT_EQ(regular_clique_requests.size(), 1);

  CollectiveCliqueRequests peer_clique_requests;
  CollectiveMemoryRequests peer_memory_requests(buffer_allocations);
  Thunk::PrepareParams peer_params{&collective_params, &peer_clique_requests,
                                   &peer_memory_requests, executor,
                                   &buffer_allocations};
  auto peer_status =
      make_thunk(/*skip_collective_clique=*/true)->Prepare(peer_params);
  ASSERT_TRUE(peer_status.ok()) << peer_status;
  EXPECT_EQ(peer_clique_requests.size(), 0);

  executor->Deallocate(&device_buffer);
}

}  // namespace
}  // namespace xla::gpu
