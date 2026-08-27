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

#include "xla/backends/gpu/autotuner/nvfp4_decode_dot.h"

#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/substitute.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/autotuner/backend_config.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/nvptx_compiler.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

// A claimed decode projection with the weight on M: $0 weight rows, $1 batch.
constexpr absl::string_view kClaimedFusion = R"(
  HloModule m
  nvfp4_decode_dot_d {
    p0 = f4e2m1fn[$0,4096] parameter(0)
    p1 = f4e2m1fn[$1,4096] parameter(1)
    p2 = f8e4m3fn[$0,256] parameter(2)
    p3 = f8e4m3fn[$1,256] parameter(3)
    ROOT d = bf16[$0,$1] scaled-dot(p0, p1, p2, p3),
        lhs_contracting_dims={1}, rhs_contracting_dims={1}
  }
  ENTRY main {
    a = f4e2m1fn[$0,4096] parameter(0)
    b = f4e2m1fn[$1,4096] parameter(1)
    c = f8e4m3fn[$0,256] parameter(2)
    e = f8e4m3fn[$1,256] parameter(3)
    ROOT f = bf16[$0,$1] fusion(a, b, c, e), kind=kCustom,
        calls=nvfp4_decode_dot_d,
        backend_config={"fusion_backend_config":{"kind":"__triton_nested_gemm_fusion"}}
  })";

class Nvfp4DecodeDotBackendTest : public HloHardwareIndependentTestBase {
 protected:
  Nvfp4DecodeDotBackendTest()
      : stream_executor_(PlatformUtil::GetDefaultPlatform()
                             .value()
                             ->ExecutorForDevice(0)
                             .value()),
        target_config_(stream_executor_),
        debug_options_(
            HloHardwareIndependentTestBase::GetDebugOptionsForTest()) {
    debug_options_.set_xla_gpu_experimental_scaled_dot_with_tile_ir(true);
    triton_ = std::make_unique<Nvfp4DecodeDotBackend>(
        &debug_options_, &compiler_, &target_config_, &mlir_context_,
        Nvfp4DecodeDotBackend::Rung::kTriton);
    tile_ir_ = std::make_unique<Nvfp4DecodeDotBackend>(
        &debug_options_, &compiler_, &target_config_, &mlir_context_,
        Nvfp4DecodeDotBackend::Rung::kTileIr);
  }

  const HloInstruction& RootOf(HloModule& module) {
    return *module.entry_computation()->root_instruction();
  }

  bool IsSm103() const {
    const se::CudaComputeCapability cc =
        target_config_.device_description.cuda_compute_capability();
    return cc.major == se::CudaComputeCapability::kBlackwell && cc.minor == 3;
  }
  bool IsSm120() const {
    return target_config_.device_description.cuda_compute_capability().major ==
           se::CudaComputeCapability::kBlackwell_12;
  }

  // The batch side of a config's output tile, for a weight-on-M fusion.
  static int64_t BatchTile(const BackendConfig& config, bool tile_ir) {
    const TileIrFusionConfig& c =
        tile_ir ? config.nvfp4_decode_dot_tile_ir() : config.nvfp4_decode_dot();
    return c.block_level_fusion_config().output_tiles(0).sizes(1);
  }

  NVPTXCompiler compiler_;
  se::StreamExecutor* stream_executor_;
  Compiler::GpuTargetConfig target_config_;
  DebugOptions debug_options_;
  mlir::MLIRContext mlir_context_;
  std::unique_ptr<Nvfp4DecodeDotBackend> triton_;
  std::unique_ptr<Nvfp4DecodeDotBackend> tile_ir_;
};

TEST_F(Nvfp4DecodeDotBackendTest, TileIrRungTilesTheBatchExactlyOnSm103) {
  if (!IsSm103()) {
    GTEST_SKIP() << "The Tile IR rung's batch width is an sm_103 tileiras limit.";
  }
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           absl::Substitute(kClaimedFusion, 5120, 16)));
  EXPECT_TRUE(tile_ir_->IsSupported(RootOf(*module)));
  TF_ASSERT_OK_AND_ASSIGN(auto configs,
                          tile_ir_->GetSupportedConfigs(RootOf(*module)));
  ASSERT_FALSE(configs.empty());
  for (const auto& config : configs) {
    ASSERT_TRUE(config->has_nvfp4_decode_dot_tile_ir());
    EXPECT_EQ(BatchTile(*config, /*tile_ir=*/true), 16);
    EXPECT_GE(
        config->nvfp4_decode_dot_tile_ir().block_level_fusion_config().output_tiles(0).sizes(0),
        128);
  }
}

TEST_F(Nvfp4DecodeDotBackendTest, TileIrRungDeclinesABatchWiderThanTileirasRuns) {
  if (!IsSm103()) {
    GTEST_SKIP() << "The Tile IR rung's batch width is an sm_103 tileiras limit.";
  }
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           absl::Substitute(kClaimedFusion, 5120, 32)));
  TF_ASSERT_OK_AND_ASSIGN(auto configs,
                          tile_ir_->GetSupportedConfigs(RootOf(*module)));
  EXPECT_TRUE(configs.empty());
}

TEST_F(Nvfp4DecodeDotBackendTest, TritonRungStillSearchesWiderBatchTiles) {
  if (!IsSm103() && !IsSm120()) {
    GTEST_SKIP() << "The arm claims on Blackwell only.";
  }
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           absl::Substitute(kClaimedFusion, 5120, 16)));
  TF_ASSERT_OK_AND_ASSIGN(auto configs,
                          triton_->GetSupportedConfigs(RootOf(*module)));
  ASSERT_FALSE(configs.empty());
  bool wider = false;
  for (const auto& config : configs) {
    ASSERT_TRUE(config->has_nvfp4_decode_dot());
    wider |= BatchTile(*config, /*tile_ir=*/false) > 16;
  }
  EXPECT_TRUE(wider);
}

TEST_F(Nvfp4DecodeDotBackendTest, TileIrRungOffersNothingOnSm120) {
  if (!IsSm120()) {
    GTEST_SKIP() << "sm_120 has no Tile IR rung of its own.";
  }
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(
                                           absl::Substitute(kClaimedFusion, 5120, 16)));
  EXPECT_FALSE(tile_ir_->IsSupported(RootOf(*module)));
  TF_ASSERT_OK_AND_ASSIGN(auto configs,
                          tile_ir_->GetSupportedConfigs(RootOf(*module)));
  EXPECT_TRUE(configs.empty());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
