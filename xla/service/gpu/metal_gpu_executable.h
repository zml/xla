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

#ifndef XLA_SERVICE_GPU_METAL_GPU_EXECUTABLE_H_
#define XLA_SERVICE_GPU_METAL_GPU_EXECUTABLE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/executable.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/shape.h"

namespace xla {
namespace gpu {

struct MetalMatmulConfig {
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  bool relu = false;
};

absl::StatusOr<MetalMatmulConfig> MatchMetalMatmul(const HloModule& module);

absl::StatusOr<std::vector<uint8_t>> CompileMetalMatmulAirToMetallib();

absl::StatusOr<std::unique_ptr<Executable>> BuildMetalElementwiseExecutable(
    std::shared_ptr<HloModule> module);

absl::StatusOr<std::unique_ptr<Executable>> BuildMetalReductionExecutable(
    std::shared_ptr<HloModule> module);

absl::StatusOr<std::unique_ptr<Executable>> BuildMetalConvertExecutable(
    std::shared_ptr<HloModule> module);

class MetalMatmulExecutable final : public Executable {
 public:
  MetalMatmulExecutable(std::shared_ptr<HloModule> module,
                        MetalMatmulConfig config,
                        std::vector<uint8_t> metallib);

  Shape result_shape() const override;

  absl::StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments) override;

 private:
  MetalMatmulConfig config_;
  Shape result_shape_;
  std::string kernel_name_;
  std::vector<uint8_t> metallib_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_METAL_GPU_EXECUTABLE_H_
