/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/codegen/triton/compilation_pipeline.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/sycl/oneapi_compute_capability.h"

namespace xla {
namespace gpu {

using ::testing::Contains;

constexpr uint32_t kBmgIpVersion = (0x14u << 22) | (0x2u << 14);
constexpr uint32_t kUnknownOneapiIpVersion = 0x99u << 22;

TEST(CompilationPipelineTest, ContainsUnswitchLoopsCompositePass) {
  mlir::MLIRContext ctx;
  mlir::PassManager pm(&ctx);

  CreateTritonXlaPipeline(&pm, stream_executor::CudaComputeCapability(),
                          /*rewrite_int4=*/false, /*allow_tma=*/true,
                          /*num_stages=*/1,
                          /*warp_specialization_allowed=*/true,
                          /*enable_pdl=*/false);

  std::vector<std::string> pass_names;
  for (const mlir::Pass& pass : pm.getPasses()) {
    pass_names.push_back(pass.getName().str());
  }
  ASSERT_THAT(pass_names, Contains("TritonXLAUnswitchLoopsComposite"));
}

TEST(CompilationPipelineTest, SelectsOneApiXpuPipeline) {
  mlir::MLIRContext ctx;
  mlir::PassManager pm(&ctx);
  stream_executor::DeviceDescription device_info;
  device_info.set_oneapi_compute_capability(kBmgIpVersion);
  device_info.set_threads_per_warp(16);

  absl::Status status = CreateTritonPipeline(&pm, device_info,
                                             /*num_warps=*/4,
                                             /*num_ctas=*/1,
                                             /*num_stages=*/3);

  EXPECT_TRUE(status.ok()) << status;
}

TEST(CompilationPipelineTest, RejectsUnknownOneApiArchitecture) {
  mlir::MLIRContext ctx;
  mlir::PassManager pm(&ctx);
  stream_executor::DeviceDescription device_info;
  device_info.set_oneapi_compute_capability(kUnknownOneapiIpVersion);
  device_info.set_threads_per_warp(16);

  absl::Status status = CreateTritonPipeline(&pm, device_info,
                                             /*num_warps=*/4,
                                             /*num_ctas=*/1,
                                             /*num_stages=*/3);

  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
}

}  // namespace gpu
}  // namespace xla
