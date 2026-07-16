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

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/musa/musa_executor.h"
#include "xla/stream_executor/musa/musa_platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace stream_executor::musa {
namespace {

std::vector<uint8_t> ReadExternalMubin(const char* environment_variable) {
  const char* path = std::getenv(environment_variable);
  if (path == nullptr || path[0] == '\0') return {};
  std::ifstream input(path, std::ios::binary);
  if (!input) return {};
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(input),
                              std::istreambuf_iterator<char>());
}

TEST(MusaLaunchSmokeTest, ExecutesKnownMubinAndStreamPrimitivesOnS80) {
  std::vector<uint8_t> image = ReadExternalMubin("MUSA_TEST_MUBIN");
  if (image.empty()) {
    GTEST_SKIP() << "Set MUSA_TEST_MUBIN to the external add_one MUBIN";
  }

  MusaExecutor executor(/*platform=*/nullptr, /*device_ordinal=*/0);
  ASSERT_TRUE(executor.Init().ok());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> producer,
                          executor.CreateStream(/*priority=*/std::nullopt));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> consumer,
                          executor.CreateStream(/*priority=*/std::nullopt));

  DeviceAddressBase input = executor.Allocate(4 * sizeof(float), 0);
  DeviceAddressBase output = executor.Allocate(4 * sizeof(float), 0);
  DeviceAddressBase pattern_buffer = executor.Allocate(4 * sizeof(uint32_t), 0);
  DeviceAddressBase timer_buffer = executor.Allocate(4 * 1024 * 1024, 0);
  ASSERT_FALSE(input.is_null());
  ASSERT_FALSE(output.is_null());
  ASSERT_FALSE(pattern_buffer.is_null());
  ASSERT_FALSE(timer_buffer.is_null());

  std::array<float, 4> host_input = {1.0f, 2.0f, 3.0f, 4.0f};
  std::array<float, 4> host_output = {};
  TF_ASSERT_OK(producer->Memcpy(&input, host_input.data(), sizeof(host_input)));

  KernelLoaderSpec kernel_spec = KernelLoaderSpec::CreateMusaMubinInMemorySpec(
      image, "add_one", /*arity=*/3);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Kernel> kernel,
                          executor.LoadKernel(kernel_spec));
  TF_ASSERT_OK_AND_ASSIGN(int32_t occupancy,
                          kernel->GetMaxOccupiedBlocksPerCore(ThreadDim(4), 0));
  EXPECT_GT(occupancy, 0);

  KernelArgsPackedArray arguments(/*num_args=*/3);
  arguments.add_argument(input);
  arguments.add_argument(output);
  arguments.add_argument(int64_t{4});
  arguments.add_shared_bytes(128);
  TF_ASSERT_OK(
      kernel->Launch(ThreadDim(4), BlockDim(1), producer.get(), arguments));
  TF_ASSERT_OK(
      producer->Memcpy(host_output.data(), output, sizeof(host_output)));

  std::atomic<bool> callback_ran = false;
  TF_ASSERT_OK(producer->DoHostCallbackWithStatus([&callback_ran] {
    callback_ran = true;
    return absl::OkStatus();
  }));
  TF_ASSERT_OK(producer->BlockHostUntilDone());
  EXPECT_TRUE(callback_ran.load());
  EXPECT_EQ(host_output, (std::array<float, 4>{2.0f, 3.0f, 4.0f, 5.0f}));

  constexpr uint32_t kPattern = 0x12345678;
  TF_ASSERT_OK(
      producer->Memset32(&pattern_buffer, kPattern, 4 * sizeof(uint32_t)));
  TF_ASSERT_OK(consumer->WaitFor(producer.get()));
  std::array<uint32_t, 4> patterns = {};
  TF_ASSERT_OK(
      consumer->Memcpy(patterns.data(), pattern_buffer, sizeof(patterns)));
  TF_ASSERT_OK(consumer->BlockHostUntilDone());
  EXPECT_EQ(patterns,
            (std::array<uint32_t, 4>{kPattern, kPattern, kPattern, kPattern}));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<EventBasedTimer> timer,
      producer->CreateEventBasedTimer(/*use_delay_kernel=*/false));
  TF_ASSERT_OK(producer->MemZero(&timer_buffer, timer_buffer.size()));
  TF_ASSERT_OK_AND_ASSIGN(absl::Duration elapsed, timer->GetElapsedDuration());
  EXPECT_GT(elapsed, absl::ZeroDuration());

  absl::Status cluster_status = kernel->Launch(
      ThreadDim(4), BlockDim(1), ClusterDim(1), producer.get(), arguments);
  EXPECT_EQ(cluster_status.code(), absl::StatusCode::kUnimplemented);
  kernel->set_use_pdl(true);
  absl::Status pdl_status =
      kernel->Launch(ThreadDim(4), BlockDim(1), producer.get(), arguments);
  EXPECT_EQ(pdl_status.code(), absl::StatusCode::kUnimplemented);
  kernel->set_use_pdl(false);

  MultiModuleLoaderSpec module_spec;
  module_spec.AddMusaMubinInMemory(image);
  TF_ASSERT_OK_AND_ASSIGN(ModuleHandle module,
                          executor.LoadModule(module_spec));
  absl::Status missing_symbol =
      executor.GetSymbol("__xla_missing_global", module).status();
  EXPECT_EQ(missing_symbol.code(), absl::StatusCode::kNotFound);
  EXPECT_TRUE(executor.UnloadModule(module));

  kernel.reset();
  timer.reset();
  producer.reset();
  consumer.reset();
  executor.Deallocate(&timer_buffer);
  executor.Deallocate(&pattern_buffer);
  executor.Deallocate(&output);
  executor.Deallocate(&input);
}

struct AggregateArgument {
  int32_t integer;
  float real;
};

absl::Status RunKernelLifetimeStressWorker(
    MusaExecutor* executor, Stream* stream, const std::vector<uint8_t>& image,
    DeviceAddressBase input, DeviceAddressBase output, int worker) {
  const std::array<float, 4> host_input = {
      static_cast<float>(worker), static_cast<float>(worker + 1),
      static_cast<float>(worker + 2), static_cast<float>(worker + 3)};
  TF_RETURN_IF_ERROR(
      stream->Memcpy(&input, host_input.data(), sizeof(host_input)));

  for (int iteration = 0; iteration < 16; ++iteration) {
    KernelLoaderSpec spec = KernelLoaderSpec::CreateMusaMubinInMemorySpec(
        image, "add_one", /*arity=*/3);
    TF_ASSIGN_OR_RETURN(std::unique_ptr<Kernel> kernel,
                        executor->LoadKernel(spec));
    KernelArgsPackedArray args(/*num_args=*/3);
    args.add_argument(input);
    args.add_argument(output);
    args.add_argument(int64_t{4});
    TF_RETURN_IF_ERROR(kernel->Launch(ThreadDim(4), BlockDim(1), stream, args));

    // Destroy the last strong kernel reference while its launch may still be
    // queued. MusaKernel must retain its module through completion.
    kernel.reset();
  }

  std::array<float, 4> host_output = {};
  TF_RETURN_IF_ERROR(
      stream->Memcpy(host_output.data(), output, sizeof(host_output)));
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  for (int i = 0; i < host_output.size(); ++i) {
    if (host_output[i] != host_input[i] + 1.0f) {
      return absl::InternalError("concurrent MUSA launch produced bad output");
    }
  }
  return absl::OkStatus();
}

TEST(MusaLaunchSmokeTest, ExtendedStreamExecutorConformanceOnS80) {
  std::vector<uint8_t> image = ReadExternalMubin("MUSA_TEST_CONFORMANCE_MUBIN");
  if (image.empty()) {
    GTEST_SKIP() << "Set MUSA_TEST_CONFORMANCE_MUBIN to launch_fixture.mubin";
  }

  MusaPlatform platform;
  MusaExecutor executor(&platform, /*device_ordinal=*/0,
                        /*callback_poll_interval=*/absl::Milliseconds(10));
  TF_ASSERT_OK(executor.Init());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Stream> stream,
                          executor.CreateStream(/*priority=*/std::nullopt));

  MultiModuleLoaderSpec module_spec;
  module_spec.AddMusaMubinInMemory(image);
  TF_ASSERT_OK_AND_ASSIGN(ModuleHandle module,
                          executor.LoadModule(module_spec));
  TF_ASSERT_OK_AND_ASSIGN(DeviceAddressBase global,
                          executor.GetSymbol("launch_fixture_global", module));
  TF_ASSERT_OK_AND_ASSIGN(
      DeviceAddressBase constant,
      executor.GetSymbol("launch_fixture_constant", module));
  EXPECT_EQ(global.size(), 4 * sizeof(uint32_t));
  EXPECT_EQ(constant.size(), 4 * sizeof(uint32_t));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(global.opaque()) % 16, 0);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(constant.opaque()) % 16, 0);

  std::array<uint32_t, 4> direct_global = {};
  std::array<uint32_t, 4> direct_constant = {};
  TF_ASSERT_OK(executor.SynchronousMemcpy(direct_global.data(), global,
                                          sizeof(direct_global)));
  TF_ASSERT_OK(executor.SynchronousMemcpy(direct_constant.data(), constant,
                                          sizeof(direct_constant)));
  EXPECT_EQ(direct_global, (std::array<uint32_t, 4>{0x10203040u, 0x50607080u,
                                                    0x90a0b0c0u, 0xd0e0f000u}));
  EXPECT_EQ(direct_constant, (std::array<uint32_t, 4>{3u, 5u, 7u, 11u}));

  KernelLoaderSpec no_args_spec = KernelLoaderSpec::CreateMusaMubinInMemorySpec(
      image, "no_args", /*arity=*/0);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Kernel> no_args,
                          executor.LoadKernel(no_args_spec));
  KernelArgsPackedArray no_arguments(/*num_args=*/0);
  TF_ASSERT_OK(
      no_args->Launch(ThreadDim(1), BlockDim(1), stream.get(), no_arguments));
  TF_ASSERT_OK(stream->BlockHostUntilDone());
  TF_ASSERT_OK(executor.SynchronousMemcpy(direct_global.data(), global,
                                          sizeof(direct_global)));
  EXPECT_EQ(direct_global[0], 0x10203041u);

  DeviceAddressBase global_output =
      executor.Allocate(4 * sizeof(uint32_t), /*memory_space=*/0);
  DeviceAddressBase constant_output =
      executor.Allocate(4 * sizeof(uint32_t), /*memory_space=*/0);
  ASSERT_FALSE(global_output.is_null());
  ASSERT_FALSE(constant_output.is_null());
  KernelLoaderSpec read_symbols_spec =
      KernelLoaderSpec::CreateMusaMubinInMemorySpec(image, "read_symbols",
                                                    /*arity=*/2);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Kernel> read_symbols,
                          executor.LoadKernel(read_symbols_spec));
  KernelArgsPackedArray symbol_arguments(/*num_args=*/2);
  symbol_arguments.add_argument(global_output);
  symbol_arguments.add_argument(constant_output);
  TF_ASSERT_OK(read_symbols->Launch(ThreadDim(4), BlockDim(1), stream.get(),
                                    symbol_arguments));
  std::array<uint32_t, 4> copied_global = {};
  std::array<uint32_t, 4> copied_constant = {};
  TF_ASSERT_OK(stream->Memcpy(copied_global.data(), global_output,
                              sizeof(copied_global)));
  TF_ASSERT_OK(stream->Memcpy(copied_constant.data(), constant_output,
                              sizeof(copied_constant)));
  TF_ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_EQ(copied_global, direct_global);
  EXPECT_EQ(copied_constant, direct_constant);

  DeviceAddressBase integer_output =
      executor.Allocate(4 * sizeof(int64_t), /*memory_space=*/0);
  DeviceAddressBase narrow_output =
      executor.Allocate(2 * sizeof(uint16_t), /*memory_space=*/0);
  DeviceAddressBase floating_output =
      executor.Allocate(2 * sizeof(double), /*memory_space=*/0);
  ASSERT_FALSE(integer_output.is_null());
  ASSERT_FALSE(narrow_output.is_null());
  ASSERT_FALSE(floating_output.is_null());
  KernelLoaderSpec scalar_spec = KernelLoaderSpec::CreateMusaMubinInMemorySpec(
      image, "scalar_mix", /*arity=*/11);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Kernel> scalar,
                          executor.LoadKernel(scalar_spec));
  constexpr int32_t kSigned32 = -1234567;
  constexpr uint32_t kUnsigned32 = 4000000000u;
  constexpr int64_t kSigned64 = -0x123456789abcdefLL;
  constexpr uint64_t kUnsigned64 = 0x123456789abcdef0ULL;
  constexpr uint16_t kF16OneBits = 0x3c00;
  constexpr uint16_t kBf16OneBits = 0x3f80;
  constexpr float kFloat32 = 3.25f;
  constexpr double kFloat64 = -9.5;
  KernelArgsPackedArray scalar_arguments(/*num_args=*/11);
  scalar_arguments.add_argument(kSigned32);
  scalar_arguments.add_argument(kUnsigned32);
  scalar_arguments.add_argument(kSigned64);
  scalar_arguments.add_argument(kUnsigned64);
  scalar_arguments.add_argument(kF16OneBits);
  scalar_arguments.add_argument(kBf16OneBits);
  scalar_arguments.add_argument(kFloat32);
  scalar_arguments.add_argument(kFloat64);
  scalar_arguments.add_argument(integer_output);
  scalar_arguments.add_argument(narrow_output);
  scalar_arguments.add_argument(floating_output);
  TF_ASSERT_OK(scalar->Launch(ThreadDim(1), BlockDim(1), stream.get(),
                              scalar_arguments));
  std::array<int64_t, 4> integers = {};
  std::array<uint16_t, 2> narrows = {};
  std::array<double, 2> floating = {};
  TF_ASSERT_OK(
      stream->Memcpy(integers.data(), integer_output, sizeof(integers)));
  TF_ASSERT_OK(stream->Memcpy(narrows.data(), narrow_output, sizeof(narrows)));
  TF_ASSERT_OK(
      stream->Memcpy(floating.data(), floating_output, sizeof(floating)));
  TF_ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_EQ(integers,
            (std::array<int64_t, 4>{kSigned32, kUnsigned32, kSigned64,
                                    static_cast<int64_t>(kUnsigned64)}));
  EXPECT_EQ(narrows, (std::array<uint16_t, 2>{kF16OneBits, kBf16OneBits}));
  EXPECT_DOUBLE_EQ(floating[0], static_cast<double>(kFloat32));
  EXPECT_DOUBLE_EQ(floating[1], kFloat64);

  DeviceAddressBase aggregate_integer =
      executor.Allocate(sizeof(int32_t), /*memory_space=*/0);
  DeviceAddressBase aggregate_float =
      executor.Allocate(sizeof(float), /*memory_space=*/0);
  ASSERT_FALSE(aggregate_integer.is_null());
  ASSERT_FALSE(aggregate_float.is_null());
  KernelLoaderSpec aggregate_spec =
      KernelLoaderSpec::CreateMusaMubinInMemorySpec(image, "aggregate_arg",
                                                    /*arity=*/3);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Kernel> aggregate,
                          executor.LoadKernel(aggregate_spec));
  const AggregateArgument aggregate_value{0x1234567, -7.75f};
  KernelArgsPackedArray aggregate_arguments(/*num_args=*/3);
  aggregate_arguments.add_argument(aggregate_value);
  aggregate_arguments.add_argument(aggregate_integer);
  aggregate_arguments.add_argument(aggregate_float);
  TF_ASSERT_OK(aggregate->Launch(ThreadDim(1), BlockDim(1), stream.get(),
                                 aggregate_arguments));
  int32_t aggregate_integer_host = 0;
  float aggregate_float_host = 0.0f;
  TF_ASSERT_OK(stream->Memcpy(&aggregate_integer_host, aggregate_integer,
                              sizeof(aggregate_integer_host)));
  TF_ASSERT_OK(stream->Memcpy(&aggregate_float_host, aggregate_float,
                              sizeof(aggregate_float_host)));
  TF_ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_EQ(aggregate_integer_host, aggregate_value.integer);
  EXPECT_FLOAT_EQ(aggregate_float_host, aggregate_value.real);

  DeviceAddressBase dimension_output =
      executor.Allocate(12 * sizeof(uint32_t), /*memory_space=*/0);
  ASSERT_FALSE(dimension_output.is_null());
  KernelLoaderSpec dimensions_spec =
      KernelLoaderSpec::CreateMusaMubinInMemorySpec(image, "record_dimensions",
                                                    /*arity=*/1);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Kernel> dimensions,
                          executor.LoadKernel(dimensions_spec));
  const std::array<std::pair<BlockDim, ThreadDim>, 3> dimension_cases = {
      std::pair{BlockDim(7, 1, 1), ThreadDim(16, 1, 1)},
      std::pair{BlockDim(3, 2, 1), ThreadDim(8, 4, 1)},
      std::pair{BlockDim(2, 3, 4), ThreadDim(4, 2, 2)}};
  for (const auto& [blocks, threads] : dimension_cases) {
    TF_ASSERT_OK(stream->MemZero(&dimension_output, dimension_output.size()));
    KernelArgsPackedArray dimension_arguments(/*num_args=*/1);
    dimension_arguments.add_argument(dimension_output);
    TF_ASSERT_OK(
        dimensions->Launch(threads, blocks, stream.get(), dimension_arguments));
    std::array<uint32_t, 12> recorded = {};
    TF_ASSERT_OK(
        stream->Memcpy(recorded.data(), dimension_output, sizeof(recorded)));
    TF_ASSERT_OK(stream->BlockHostUntilDone());
    EXPECT_EQ(recorded[6], blocks.x);
    EXPECT_EQ(recorded[7], blocks.y);
    EXPECT_EQ(recorded[8], blocks.z);
    EXPECT_EQ(recorded[9], threads.x);
    EXPECT_EQ(recorded[10], threads.y);
    EXPECT_EQ(recorded[11], threads.z);
  }

  KernelArgsPackedArray dimension_arguments(/*num_args=*/1);
  dimension_arguments.add_argument(dimension_output);
  EXPECT_EQ(dimensions
                ->Launch(ThreadDim(0, 1, 1), BlockDim(1), stream.get(),
                         dimension_arguments)
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(dimensions
                ->Launch(ThreadDim(1), BlockDim(0, 1, 1), stream.get(),
                         dimension_arguments)
                .code(),
            absl::StatusCode::kInvalidArgument);
  const DeviceDescription& description = executor.GetDeviceDescription();
  const uint64_t max_threads = description.threads_per_block_limit();
  TF_ASSERT_OK(dimensions->Launch(ThreadDim(max_threads, 1, 1), BlockDim(1),
                                  stream.get(), dimension_arguments));
  EXPECT_EQ(dimensions
                ->Launch(ThreadDim(max_threads, 2, 1), BlockDim(1),
                         stream.get(), dimension_arguments)
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(dimensions
                ->Launch(ThreadDim(description.thread_dim_limit().x + 1, 1, 1),
                         BlockDim(1), stream.get(), dimension_arguments)
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(dimensions
                ->Launch(ThreadDim(1),
                         BlockDim(description.block_dim_limit().x + 1, 1, 1),
                         stream.get(), dimension_arguments)
                .code(),
            absl::StatusCode::kInvalidArgument);
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  constexpr int kReverseCount = 8;
  DeviceAddressBase reverse_input =
      executor.Allocate(kReverseCount * sizeof(float), /*memory_space=*/0);
  DeviceAddressBase reverse_output =
      executor.Allocate(kReverseCount * sizeof(float), /*memory_space=*/0);
  ASSERT_FALSE(reverse_input.is_null());
  ASSERT_FALSE(reverse_output.is_null());
  const std::array<float, kReverseCount> reverse_host_input = {1, 2, 3, 4,
                                                               5, 6, 7, 8};
  TF_ASSERT_OK(stream->Memcpy(&reverse_input, reverse_host_input.data(),
                              sizeof(reverse_host_input)));
  KernelLoaderSpec reverse_spec = KernelLoaderSpec::CreateMusaMubinInMemorySpec(
      image, "reverse_shared", /*arity=*/3);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Kernel> reverse,
                          executor.LoadKernel(reverse_spec));
  KernelArgsPackedArray reverse_arguments(/*num_args=*/3);
  reverse_arguments.add_argument(reverse_input);
  reverse_arguments.add_argument(reverse_output);
  reverse_arguments.add_argument(int64_t{kReverseCount});
  reverse_arguments.add_shared_bytes(4 * sizeof(float));
  TF_ASSERT_OK(reverse->Launch(ThreadDim(4), BlockDim(2), stream.get(),
                               reverse_arguments));
  std::array<float, kReverseCount> reverse_host_output = {};
  TF_ASSERT_OK(stream->Memcpy(reverse_host_output.data(), reverse_output,
                              sizeof(reverse_host_output)));
  TF_ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_EQ(reverse_host_output,
            (std::array<float, kReverseCount>{4, 3, 2, 1, 8, 7, 6, 5}));

  std::promise<absl::Status> callback_error;
  std::future<absl::Status> callback_error_future = callback_error.get_future();
  TF_ASSERT_OK(stream->DoHostCallbackWithStatus(
      [] { return absl::InternalError("intentional callback failure"); },
      [&callback_error](absl::Status status) {
        callback_error.set_value(std::move(status));
      }));
  TF_ASSERT_OK(stream->BlockHostUntilDone());
  absl::Status forwarded_error = callback_error_future.get();
  EXPECT_EQ(forwarded_error.code(), absl::StatusCode::kInternal);
  EXPECT_EQ(forwarded_error.message(), "intentional callback failure");

  DeviceAddressBase thunk_input =
      executor.Allocate(sizeof(int32_t), /*memory_space=*/0);
  DeviceAddressBase thunk_output =
      executor.Allocate(sizeof(int32_t), /*memory_space=*/0);
  ASSERT_FALSE(thunk_input.is_null());
  ASSERT_FALSE(thunk_output.is_null());
  const int32_t thunk_input_host = 41;
  TF_ASSERT_OK(stream->Memcpy(&thunk_input, &thunk_input_host,
                              sizeof(thunk_input_host)));
  xla::BufferAllocation thunk_input_allocation(
      /*index=*/0, /*size=*/sizeof(int32_t), /*color=*/0);
  xla::BufferAllocation thunk_output_allocation(
      /*index=*/1, /*size=*/sizeof(int32_t), /*color=*/0);
  xla::emitters::KernelArgument thunk_input_argument(
      xla::ShapeUtil::MakeShape(xla::S32, {1}),
      xla::BufferAllocation::Slice(&thunk_input_allocation, 0,
                                   sizeof(int32_t)));
  xla::emitters::KernelArgument thunk_output_argument(
      xla::ShapeUtil::MakeShape(xla::S32, {1}),
      xla::BufferAllocation::Slice(&thunk_output_allocation, 0,
                                   sizeof(int32_t)));
  thunk_input_argument.set_written(false);
  thunk_output_argument.set_written(true);
  xla::gpu::KernelThunk thunk(
      xla::gpu::Thunk::ThunkInfo(), "copy_plus_one_i32",
      xla::emitters::KernelArguments(
          {thunk_input_argument, thunk_output_argument}),
      xla::gpu::LaunchDimensions(BlockDim(1), ThreadDim(1)),
      /*cluster_dim=*/std::nullopt, /*shmem_bytes=*/0, gpu::TmaMetadata());
  xla::gpu::BufferAllocations thunk_allocations({thunk_input, thunk_output},
                                                executor.device_ordinal(),
                                                /*memory_allocator=*/nullptr);
  xla::gpu::Thunk::ExecutableSource thunk_source;
  thunk_source.binary = image;
  xla::gpu::Thunk::InitializeParams initialize_params;
  initialize_params.executor = &executor;
  initialize_params.src = thunk_source;
  initialize_params.buffer_allocations = &thunk_allocations;
  initialize_params.stream = stream.get();
  TF_ASSERT_OK(thunk.Initialize(initialize_params));
  xla::ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream.get());
  xla::gpu::Thunk::ExecuteParams execute_params =
      xla::gpu::Thunk::ExecuteParams::Create(
          run_options, thunk_allocations, stream.get(),
          /*command_buffer_trace_stream=*/nullptr,
          /*collective_params=*/nullptr, /*collective_cliques=*/nullptr,
          /*collective_memory=*/nullptr);
  TF_ASSERT_OK(thunk.ExecuteOnStream(execute_params));
  int32_t thunk_output_host = 0;
  TF_ASSERT_OK(stream->Memcpy(&thunk_output_host, thunk_output,
                              sizeof(thunk_output_host)));
  TF_ASSERT_OK(stream->BlockHostUntilDone());
  EXPECT_EQ(thunk_output_host, 42);

  constexpr int kStressWorkers = 4;
  std::array<std::unique_ptr<Stream>, kStressWorkers> stress_streams;
  std::array<DeviceAddressBase, kStressWorkers> stress_inputs;
  std::array<DeviceAddressBase, kStressWorkers> stress_outputs;
  for (int i = 0; i < kStressWorkers; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(stress_streams[i],
                            executor.CreateStream(/*priority=*/std::nullopt));
    stress_inputs[i] = executor.Allocate(4 * sizeof(float), /*memory_space=*/0);
    stress_outputs[i] =
        executor.Allocate(4 * sizeof(float), /*memory_space=*/0);
    ASSERT_FALSE(stress_inputs[i].is_null());
    ASSERT_FALSE(stress_outputs[i].is_null());
  }
  std::array<absl::Status, kStressWorkers> stress_statuses;
  std::array<std::thread, kStressWorkers> workers;
  for (int i = 0; i < kStressWorkers; ++i) {
    workers[i] = std::thread([&, i] {
      stress_statuses[i] = RunKernelLifetimeStressWorker(
          &executor, stress_streams[i].get(), image, stress_inputs[i],
          stress_outputs[i], i);
    });
  }
  for (std::thread& worker : workers) worker.join();
  for (const absl::Status& status : stress_statuses) TF_EXPECT_OK(status);

  for (int i = 0; i < kStressWorkers; ++i) {
    stress_streams[i].reset();
    executor.Deallocate(&stress_outputs[i]);
    executor.Deallocate(&stress_inputs[i]);
  }
  executor.Deallocate(&thunk_output);
  executor.Deallocate(&thunk_input);
  executor.Deallocate(&reverse_output);
  executor.Deallocate(&reverse_input);
  executor.Deallocate(&dimension_output);
  executor.Deallocate(&aggregate_float);
  executor.Deallocate(&aggregate_integer);
  executor.Deallocate(&floating_output);
  executor.Deallocate(&narrow_output);
  executor.Deallocate(&integer_output);
  executor.Deallocate(&constant_output);
  executor.Deallocate(&global_output);
  EXPECT_TRUE(executor.UnloadModule(module));
}

}  // namespace
}  // namespace stream_executor::musa
