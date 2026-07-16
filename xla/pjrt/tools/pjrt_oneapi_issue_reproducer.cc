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
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/maybe_owning_mlir_module.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using Clock = std::chrono::steady_clock;

constexpr int64_t kElements = 1 << 20;
constexpr std::array<int64_t, 1> kDimensions = {kElements};
constexpr absl::string_view kProgram = R"(
  module {
    func.func @main(%arg0: tensor<1048576xf32>) -> tensor<1048576xf32> {
      %0 = stablehlo.add %arg0, %arg0 : tensor<1048576xf32>
      return %0 : tensor<1048576xf32>
    }
  })";
const std::vector<float> kInput(kElements, 21.0f);

double Milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

double ExecuteAndWait(PjRtLoadedExecutable& executable, PjRtBuffer& input,
                      PjRtDevice& device,
                      std::unique_ptr<PjRtBuffer>* output) {
  std::array<PjRtBuffer*, 1> arguments = {&input};
  const Clock::time_point start = Clock::now();
  auto result =
      executable.ExecuteSharded(arguments, &device, ExecuteOptions());
  CHECK_OK(result);
  CHECK_EQ(result->size(), 1);
  CHECK_OK(result->front()->GetReadyFuture().Await());
  *output = std::move(result->front());
  return Milliseconds(start);
}

}  // namespace
}  // namespace xla

int main() {
  using namespace xla;

  GpuClientOptions client_options;
  client_options.allowed_devices = {0};
  client_options.allocator_config.kind = GpuAllocatorConfig::Kind::kPlatform;

  const Clock::time_point client_start = Clock::now();
  auto client = GetStreamExecutorGpuClient(client_options);
  CHECK_OK(client);
  const double client_ms = Milliseconds(client_start);
  PjRtDevice* const device = (*client)->addressable_devices().front();

  auto context = std::make_unique<mlir::MLIRContext>();
  auto module = ParseMlirModuleString(kProgram, *context);
  CHECK_OK(module);

  const Clock::time_point compile_start = Clock::now();
  auto executable = (*client)->CompileAndLoad(
      MaybeOwningMlirModule(std::move(context), std::move(*module)),
      CompileOptions());
  CHECK_OK(executable);
  const double compile_ms = Milliseconds(compile_start);

  auto input = (*client)->BufferFromHostBuffer(
      kInput.data(), F32, kDimensions, std::nullopt,
      PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, nullptr,
      *device->default_memory_space(), /*device_layout=*/nullptr);
  CHECK_OK(input);

  std::unique_ptr<PjRtBuffer> output;
  const double first_execute_ms =
      ExecuteAndWait(**executable, **input, *device, &output);
  const double second_execute_ms =
      ExecuteAndWait(**executable, **input, *device, &output);

  auto literal = output->ToLiteral().Await();
  CHECK_OK(literal);
  for (float value : (**literal).data<float>()) {
    CHECK_EQ(value, 42.0f);
  }

  std::cout << "PJRT_TIMINGS {\"client_create_ms\":" << client_ms
            << ",\"compile_ms\":" << compile_ms
            << ",\"first_execute_ms\":" << first_execute_ms
            << ",\"second_execute_ms\":" << second_execute_ms << "}\n";
  return 0;
}
