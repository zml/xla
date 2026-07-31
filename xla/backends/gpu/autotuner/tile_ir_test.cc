#include "xla/backends/gpu/autotuner/tile_ir.h"

#include <memory>

#include <gtest/gtest.h>
#include "mlir/IR/MLIRContext.h"
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

constexpr char kSameTypeScaledDotHlo[] = R"(
  fusion1 {
    lhs = f4e2m1fn[16,4096] parameter(0)
    rhs = f4e2m1fn[4096,3840] parameter(1)
    lhs_scale = f8e4m3fn[16,256] parameter(2)
    rhs_scale = f8e4m3fn[256,3840] parameter(3)
    ROOT d = f32[16,3840] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }

  ENTRY e {
    p0 = f4e2m1fn[16,4096] parameter(0)
    p1 = f4e2m1fn[4096,3840] parameter(1)
    p2 = f8e4m3fn[16,256] parameter(2)
    p3 = f8e4m3fn[256,3840] parameter(3)
    ROOT _ = f32[16,3840] fusion(p0, p1, p2, p3), kind=kCustom, calls=fusion1,
      backend_config={"fusion_backend_config":{"kind":"__scaled_gemm"}}
  })";

constexpr char kMixedTypeScaledDotHlo[] = R"(
  fusion1 {
    lhs = bf16[16,4096] parameter(0)
    rhs = f4e2m1fn[4096,3840] parameter(1)
    lhs_scale = f8e4m3fn[16,256] parameter(2)
    rhs_scale = f8e4m3fn[256,3840] parameter(3)
    ROOT d = f32[16,3840] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }

  ENTRY e {
    p0 = bf16[16,4096] parameter(0)
    p1 = f4e2m1fn[4096,3840] parameter(1)
    p2 = f8e4m3fn[16,256] parameter(2)
    p3 = f8e4m3fn[256,3840] parameter(3)
    ROOT _ = f32[16,3840] fusion(p0, p1, p2, p3), kind=kCustom, calls=fusion1,
      backend_config={"fusion_backend_config":{"kind":"__scaled_gemm"}}
  })";

constexpr char kPlainDotHlo[] = R"(
  fusion1 {
    p0 = bf16[16,4096] parameter(0)
    p1 = bf16[4096,3840] parameter(1)
    ROOT d = f32[16,3840] dot(p0, p1),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }

  ENTRY e {
    p0 = bf16[16,4096] parameter(0)
    p1 = bf16[4096,3840] parameter(1)
    ROOT _ = f32[16,3840] fusion(p0, p1), kind=kCustom, calls=fusion1,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
  })";

class TileIrBackendTest : public HloHardwareIndependentTestBase {
 protected:
  TileIrBackendTest()
      : stream_executor_(PlatformUtil::GetDefaultPlatform()
                             .value()
                             ->ExecutorForDevice(0)
                             .value()),
        target_config_(stream_executor_),
        debug_options_(
            HloHardwareIndependentTestBase::GetDebugOptionsForTest()) {
    debug_options_.set_xla_gpu_experimental_scaled_dot_with_tile_ir(true);
    backend_ = std::make_unique<TileIrBackend>(&debug_options_, &compiler_,
                                               &target_config_, &mlir_context_);
  }

  const HloInstruction& RootOf(HloModule& module) {
    return *module.entry_computation()->root_instruction();
  }

  NVPTXCompiler compiler_;
  se::StreamExecutor* stream_executor_;
  Compiler::GpuTargetConfig target_config_;
  DebugOptions debug_options_;
  mlir::MLIRContext mlir_context_;
  std::unique_ptr<TileIrBackend> backend_;
};

TEST_F(TileIrBackendTest, BidsOnASameTypeScaledDot) {
  if (!target_config_.device_description.cuda_compute_capability()
           .IsAtLeastBlackwell()) {
    GTEST_SKIP() << "Tile IR requires a block-scaled MMA (Blackwell+).";
  }
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kSameTypeScaledDotHlo));
  EXPECT_TRUE(backend_->IsSupported(RootOf(*module)));
}

TEST_F(TileIrBackendTest, DeclinesAMixedPrecisionScaledDot) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kMixedTypeScaledDotHlo));
  EXPECT_FALSE(backend_->IsSupported(RootOf(*module)));

  TF_ASSERT_OK_AND_ASSIGN(auto configs,
                          backend_->GetSupportedConfigs(RootOf(*module)));
  EXPECT_TRUE(configs.empty());
}

TEST_F(TileIrBackendTest, DeclinesAPlainDot) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kPlainDotHlo));
  EXPECT_FALSE(backend_->IsSupported(RootOf(*module)));
}

TEST_F(TileIrBackendTest, OffersNothingWhenTheFlagIsOff) {
  DebugOptions off = debug_options_;
  off.set_xla_gpu_experimental_scaled_dot_with_tile_ir(false);
  TileIrBackend backend(&off, &compiler_, &target_config_, &mlir_context_);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kSameTypeScaledDotHlo));
  EXPECT_FALSE(backend.IsSupported(RootOf(*module)));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
