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

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "xla/backends/gpu/tests/collective_ops_e2e_test_base.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_pjrt_client.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"
#include "xla/xla.pb.h"

namespace xla {
namespace {

#if TENSORFLOW_USE_ROCM
constexpr int64_t kNumReplicas = 8;
constexpr int64_t kElementsPerRank = 65536;
constexpr char kAllGatherHlo[] = R"(
HloModule all_gather

ENTRY main {
  input = bf16[65536]{0} parameter(0)
  ROOT gathered = bf16[524288]{0} all-gather(input), dimensions={0},
      replica_groups={{0,1,2,3,4,5,6,7}}
}
)";

class AllGatherE2ETestBase : public CollectiveOpsWithFlagsBase {
 public:
  AllGatherE2ETestBase()
      : CollectiveOpsWithFlagsBase(
            /*enable_async=*/false,
            /*enable_p2p_memcpy=*/false,
            /*enable_symmetric_buffer=*/false,
            /*memory_size=*/32 * kMB,
            /*collectives_memory_size=*/0) {}

 protected:
  void RunBf16EightRanks(
      gpu::CollectiveBackendConfig::CollectiveKernelStrategy
          expected_strategy,
      const char* backend_name) {
    if (device_count() < kNumReplicas) {
      GTEST_SKIP() << "Test requires eight GPUs, but found " << device_count();
    }

    ASSERT_OK_AND_ASSIGN(
        auto module,
        ParseAndReturnVerifiedModule(kAllGatherHlo,
                                     /*replica_count=*/kNumReplicas));

    std::vector<Literal> inputs;
    inputs.reserve(kNumReplicas);
    for (int64_t rank = 0; rank < kNumReplicas; ++rank) {
      std::vector<bfloat16> values(kElementsPerRank, bfloat16(rank + 1));
      inputs.push_back(LiteralUtil::CreateR1<bfloat16>(values));
    }
    std::vector<std::vector<Literal*>> arguments;
    arguments.reserve(kNumReplicas);
    for (Literal& input : inputs) {
      arguments.push_back({&input});
    }

    std::vector<bfloat16> expected_values;
    expected_values.reserve(kNumReplicas * kElementsPerRank);
    for (int64_t rank = 0; rank < kNumReplicas; ++rank) {
      expected_values.insert(expected_values.end(), kElementsPerRank,
                             bfloat16(rank + 1));
    }
    const Literal expected =
        LiteralUtil::CreateR1<bfloat16>(expected_values);

    ASSERT_OK_AND_ASSIGN(
        CollectiveOpsE2ETestBase::ExecutionResult result,
        ExecuteReplicated(std::move(module), arguments));
    ASSERT_EQ(result.results.size(), kNumReplicas);
    for (int64_t rank = 0; rank < kNumReplicas; ++rank) {
      ASSERT_TRUE(LiteralTestUtil::Equal(expected, result.results[rank]))
          << "All-gather output mismatch at rank " << rank;
    }

    // Collective-kernel scratch is double buffered. The compile-and-check run
    // initializes one half; warm the other half before measuring steady state.
    ASSERT_OK_AND_ASSIGN(
        std::vector<Literal> warmup_results,
        ExecuteReplicated(result.executable.get(), arguments));
    ASSERT_EQ(warmup_results.size(), kNumReplicas);

    constexpr int64_t kBenchmarkIterations = 200;
    const auto benchmark_start = std::chrono::steady_clock::now();
    std::vector<Literal> repeated_results;
    for (int64_t iteration = 0; iteration < kBenchmarkIterations;
         ++iteration) {
      ASSERT_OK_AND_ASSIGN(
          repeated_results,
          ExecuteReplicated(result.executable.get(), arguments));
    }
    const auto benchmark_end = std::chrono::steady_clock::now();
    const double average_microseconds =
        std::chrono::duration<double, std::micro>(benchmark_end -
                                                  benchmark_start)
            .count() /
        kBenchmarkIterations;
    std::cerr << "[ ALL-GATHER BENCHMARK ] backend=" << backend_name
              << " replicas=" << kNumReplicas
              << " bf16_elements_per_rank=" << kElementsPerRank
              << " average_us=" << average_microseconds << '\n';
    ASSERT_EQ(repeated_results.size(), kNumReplicas);
    for (int64_t rank = 0; rank < kNumReplicas; ++rank) {
      ASSERT_TRUE(LiteralTestUtil::Equal(expected, repeated_results[rank]))
          << "Repeated all-gather output mismatch at rank " << rank;
    }

    bool found_all_gather = false;
    for (const HloComputation* computation :
         result.optimized_module->computations()) {
      for (const HloInstruction* instruction : computation->instructions()) {
        if (instruction->opcode() != HloOpcode::kAllGather) {
          continue;
        }
        ASSERT_OK_AND_ASSIGN(
            gpu::GpuBackendConfig config,
            instruction->backend_config<gpu::GpuBackendConfig>());
        EXPECT_EQ(config.collective_backend_config().kernel_strategy(),
                  expected_strategy);
        found_all_gather = true;
      }
    }
    EXPECT_TRUE(found_all_gather);
  }
};

class FlyAllGatherTest : public AllGatherE2ETestBase {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions opts = CollectiveOpsWithFlagsBase::GetDebugOptionsForTest();
    opts.clear_xla_gpu_experimental_use_collective_kernels();
    opts.add_xla_gpu_experimental_use_collective_kernels(
        DebugOptions::COLLECTIVE_KERNEL_ALL_GATHER);
    opts.set_xla_gpu_enable_flydsl_fusion(true);
    opts.set_xla_gpu_flydsl_replace_triton(true);
    return opts;
  }
};

TEST_F(FlyAllGatherTest, Bf16EightRanks) {
  RunBf16EightRanks(
      gpu::CollectiveBackendConfig::KERNEL_STRATEGY_TRITON_ONE_SHOT, "fly");
}

class TritonAllGatherTest : public AllGatherE2ETestBase {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions opts = CollectiveOpsWithFlagsBase::GetDebugOptionsForTest();
    opts.clear_xla_gpu_experimental_use_collective_kernels();
    opts.add_xla_gpu_experimental_use_collective_kernels(
        DebugOptions::COLLECTIVE_KERNEL_ALL_GATHER);
    opts.set_xla_gpu_experimental_enable_tiling_propagation(true);
    opts.set_xla_gpu_enable_flydsl_fusion(false);
    opts.set_xla_gpu_flydsl_replace_triton(false);
    return opts;
  }
};

TEST_F(TritonAllGatherTest, Bf16EightRanks) {
  RunBf16EightRanks(
      gpu::CollectiveBackendConfig::KERNEL_STRATEGY_TRITON_ONE_SHOT,
      "triton");
}

class RcclAllGatherTest : public AllGatherE2ETestBase {};

TEST_F(RcclAllGatherTest, Bf16EightRanks) {
  RunBf16EightRanks(gpu::CollectiveBackendConfig::KERNEL_STRATEGY_DEFAULT,
                    "rccl");
}

enum class DeviceResidentBackend { kFly, kTriton, kRccl };

std::string_view BackendName(DeviceResidentBackend backend) {
  switch (backend) {
    case DeviceResidentBackend::kFly:
      return "fly";
    case DeviceResidentBackend::kTriton:
      return "triton";
    case DeviceResidentBackend::kRccl:
      return "rccl";
  }
}

gpu::CollectiveBackendConfig::CollectiveKernelStrategy ExpectedStrategy(
    DeviceResidentBackend backend) {
  return backend == DeviceResidentBackend::kRccl
             ? gpu::CollectiveBackendConfig::KERNEL_STRATEGY_DEFAULT
             : gpu::CollectiveBackendConfig::KERNEL_STRATEGY_TRITON_ONE_SHOT;
}

void ConfigureBackend(DeviceResidentBackend backend, DebugOptions* options) {
  options->clear_xla_gpu_experimental_use_collective_kernels();
  options->set_xla_gpu_enable_flydsl_fusion(false);
  options->set_xla_gpu_flydsl_replace_triton(false);
  options->set_xla_gpu_experimental_enable_tiling_propagation(false);
  if (backend == DeviceResidentBackend::kRccl) {
    return;
  }

  options->add_xla_gpu_experimental_use_collective_kernels(
      DebugOptions::COLLECTIVE_KERNEL_ALL_GATHER);
  if (backend == DeviceResidentBackend::kFly) {
    options->set_xla_gpu_enable_flydsl_fusion(true);
    options->set_xla_gpu_flydsl_replace_triton(true);
  } else {
    options->set_xla_gpu_experimental_enable_tiling_propagation(true);
  }
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileAllGather(
    PjRtClient& client, DeviceResidentBackend backend) {
  ASSIGN_OR_RETURN(auto module,
                   ParseAndReturnUnverifiedModule(kAllGatherHlo, {}));
  XlaComputation computation(module->ToProto());
  CompileOptions options;
  options.executable_build_options.set_num_replicas(kNumReplicas);
  ConfigureBackend(
      backend,
      options.executable_build_options.mutable_debug_options());
  return client.CompileAndLoad(computation, options);
}

void CheckStrategy(PjRtLoadedExecutable& executable,
                   DeviceResidentBackend backend) {
  ASSERT_OK_AND_ASSIGN(auto modules,
                       executable.GetExecutable()->GetHloModules());
  ASSERT_EQ(modules.size(), 1);
  bool found_all_gather = false;
  for (const HloComputation* computation : modules.front()->computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() != HloOpcode::kAllGather) {
        continue;
      }
      ASSERT_OK_AND_ASSIGN(
          gpu::GpuBackendConfig config,
          instruction->backend_config<gpu::GpuBackendConfig>());
      EXPECT_EQ(config.collective_backend_config().kernel_strategy(),
                ExpectedStrategy(backend));
      found_all_gather = true;
    }
  }
  EXPECT_TRUE(found_all_gather);
}

// Measures the PJRT execution path without copying the gathered outputs back
// to host on every iteration. The literal-based E2E harness performs eight
// 1-MiB device-to-host copies per iteration, which overwhelms the collective
// latency and is not a useful comparison between generated kernels.
TEST(DeviceResidentAllGatherBenchmark, FlyTritonRcclBf16EightRanks) {
  GpuClientOptions client_options;
  client_options.allocator_config.kind = GpuAllocatorConfig::Kind::kBFC;
  client_options.allocator_config.gpu_system_memory_size = 32 * kMB;
  client_options.use_tfrt_gpu_client = true;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                       GetXlaPjrtGpuClient(client_options));
  if (client->addressable_device_count() < kNumReplicas) {
    GTEST_SKIP() << "Test requires eight GPUs, but found "
                 << client->addressable_device_count();
  }

  std::vector<std::vector<bfloat16>> host_inputs;
  std::vector<std::unique_ptr<PjRtBuffer>> input_buffers;
  std::vector<std::vector<PjRtBuffer*>> argument_handles;
  host_inputs.reserve(kNumReplicas);
  input_buffers.reserve(kNumReplicas);
  argument_handles.reserve(kNumReplicas);
  for (int64_t rank = 0; rank < kNumReplicas; ++rank) {
    host_inputs.emplace_back(kElementsPerRank, bfloat16(rank + 1));
    PjRtDevice* device = client->addressable_devices()[rank];
    ASSERT_OK_AND_ASSIGN(PjRtMemorySpace * memory_space,
                         device->default_memory_space());
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<PjRtBuffer> input,
        client->BufferFromHostBuffer(
            host_inputs.back().data(), BF16, {kElementsPerRank},
            /*byte_strides=*/std::nullopt,
            PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
            /*on_done_with_host_buffer=*/nullptr, memory_space,
            /*device_layout=*/nullptr));
    ASSERT_OK(input->GetReadyFuture().Await());
    argument_handles.push_back({input.get()});
    input_buffers.push_back(std::move(input));
  }

  std::vector<bfloat16> expected_values;
  expected_values.reserve(kNumReplicas * kElementsPerRank);
  for (int64_t rank = 0; rank < kNumReplicas; ++rank) {
    expected_values.insert(expected_values.end(), kElementsPerRank,
                           bfloat16(rank + 1));
  }
  const Literal expected = LiteralUtil::CreateR1<bfloat16>(expected_values);

  constexpr int64_t kWarmupIterations = 10;
  constexpr int64_t kBenchmarkIterations = 200;
  for (DeviceResidentBackend backend :
       {DeviceResidentBackend::kFly, DeviceResidentBackend::kTriton,
        DeviceResidentBackend::kRccl}) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtLoadedExecutable> executable,
                         CompileAllGather(*client, backend));
    CheckStrategy(*executable, backend);

    auto execute_and_wait = [&]()
        -> absl::StatusOr<
            std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>> {
      std::optional<std::vector<Future<>>> completion_futures;
      ASSIGN_OR_RETURN(
          auto results,
          executable->Execute(absl::MakeConstSpan(argument_handles),
                              ExecuteOptions(), completion_futures));
      if (completion_futures.has_value()) {
        for (Future<>& future : *completion_futures) {
          RETURN_IF_ERROR(future.Await());
        }
      } else {
        for (auto& rank_results : results) {
          for (auto& result : rank_results) {
            RETURN_IF_ERROR(result->GetReadyFuture().Await());
          }
        }
      }
      return results;
    };

    for (int64_t iteration = 0; iteration < kWarmupIterations; ++iteration) {
      ASSERT_OK(execute_and_wait());
    }
    const auto benchmark_start = std::chrono::steady_clock::now();
    for (int64_t iteration = 0; iteration < kBenchmarkIterations;
         ++iteration) {
      ASSERT_OK(execute_and_wait());
    }
    const auto benchmark_end = std::chrono::steady_clock::now();
    const double average_microseconds =
        std::chrono::duration<double, std::micro>(benchmark_end -
                                                  benchmark_start)
            .count() /
        kBenchmarkIterations;
    std::cerr << "[ DEVICE-RESIDENT ALL-GATHER BENCHMARK ] backend="
              << BackendName(backend) << " replicas=" << kNumReplicas
              << " bf16_elements_per_rank=" << kElementsPerRank
              << " average_us=" << average_microseconds << '\n';

    ASSERT_OK_AND_ASSIGN(auto final_results, execute_and_wait());
    ASSERT_EQ(final_results.size(), kNumReplicas);
    for (int64_t rank = 0; rank < kNumReplicas; ++rank) {
      ASSERT_EQ(final_results[rank].size(), 1);
      ASSERT_OK_AND_ASSIGN(
          std::shared_ptr<Literal> actual,
          final_results[rank][0]->ToLiteral().Await());
      ASSERT_TRUE(LiteralTestUtil::Equal(expected, *actual))
          << "All-gather output mismatch at rank " << rank << " for "
          << BackendName(backend);
    }
  }
}
#endif

}  // namespace
}  // namespace xla
