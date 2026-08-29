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

#include "xla/backends/gpu/autotuner/factory.h"

#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/ascii.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiler.h"
#include "xla/service/platform_util.h"
#include "xla/shape.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/platform_object_registry.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

using autotuner::Backend;

struct FactoryTestParams {
  std::vector<Backend> names;
  int expected_num_backends;
  bool run_on_cuda = true;
  bool run_on_rocm = true;
};

class FactoryTest : public xla::HloHardwareIndependentTestBase,
                    public ::testing::WithParamInterface<FactoryTestParams> {
 protected:
  se::Platform* platform_;
  std::unique_ptr<Compiler> compiler_;
  se::StreamExecutor* stream_executor_;
  Compiler::GpuTargetConfig target_config_;
  DebugOptions debug_options_;
  stream_executor::StreamExecutorAddressAllocator allocator_;

  FactoryTest()
      : platform_(se::PlatformManager::PlatformWithName(
                      absl::AsciiStrToUpper(
                          PlatformUtil::CanonicalPlatformName("gpu").value()))
                      .value()),
        compiler_(xla::Compiler::GetForPlatform(platform_->id()).value()),
        stream_executor_(platform_->ExecutorForDevice(0).value()),
        target_config_(stream_executor_),
        allocator_(stream_executor_) {}
};

TEST_P(FactoryTest, GetCodegenBackends) {
  const auto& device = stream_executor_->GetDeviceDescription();
  bool is_cuda = device.gpu_compute_capability().IsCuda();
  bool is_rocm = device.gpu_compute_capability().IsRocm();
  if ((GetParam().run_on_cuda && is_cuda) ||
      (GetParam().run_on_rocm && is_rocm)) {
    auto& registry =
        stream_executor::PlatformObjectRegistry::GetGlobalRegistry();
    TF_ASSERT_OK_AND_ASSIGN(
        const GetCodegenBackends::Type& get_codegen_backends,
        registry.FindObject<GetCodegenBackends>(platform_->id()));
    mlir::MLIRContext mlir_context;
    AliasInfo alias_info;
    xla::RegisterSymbolicExprStorage(&mlir_context);
    std::vector<std::unique_ptr<CodegenBackend>> backends =
        get_codegen_backends(
            stream_executor_, &allocator_, &debug_options_, compiler_.get(),
            &target_config_, &alias_info, &mlir_context,
            /*shape_size_fn=*/[](const Shape&) { return 0; }, GetParam().names);
    EXPECT_EQ(backends.size(), GetParam().expected_num_backends);
  } else {
    GTEST_SKIP() << "Skipping test for platform " << platform_->id();
  }
}

TEST_P(FactoryTest, FlyFissionRewritesDecomposedDotAlgorithm) {
  if (GetParam().names !=
      std::vector<Backend>{Backend::FLY, Backend::FLY_FISSION,
                           Backend::FLY_FUSION}) {
    GTEST_SKIP() << "Only applies to the ROCm Fly backend set.";
  }
  const auto& device = stream_executor_->GetDeviceDescription();
  if (!device.gpu_compute_capability().IsRocm()) {
    GTEST_SKIP() << "Fly is a ROCm backend.";
  }

  debug_options_.set_xla_gpu_enable_flydsl_gemm(true);
  auto& registry =
      stream_executor::PlatformObjectRegistry::GetGlobalRegistry();
  TF_ASSERT_OK_AND_ASSIGN(
      const GetCodegenBackends::Type& get_codegen_backends,
      registry.FindObject<GetCodegenBackends>(platform_->id()));
  mlir::MLIRContext mlir_context;
  AliasInfo alias_info;
  xla::RegisterSymbolicExprStorage(&mlir_context);
  std::vector<std::unique_ptr<CodegenBackend>> backends =
      get_codegen_backends(
          stream_executor_, &allocator_, &debug_options_, compiler_.get(),
          &target_config_, &alias_info, &mlir_context,
          /*shape_size_fn=*/[](const Shape&) { return 0; }, GetParam().names);

  constexpr char kHlo[] = R"(
HloModule fly_bf16_x3_fission

gemm {
  lhs = f32[128,128]{1,0} parameter(0)
  rhs = f32[128,128]{0,1} parameter(1)
  ROOT dot = f32[128,128]{1,0} dot(lhs, rhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0},
      algorithm=dot_bf16_bf16_f32_x3
}

ENTRY main {
  lhs = f32[128,128]{1,0} parameter(0)
  rhs = f32[128,128]{0,1} parameter(1)
  ROOT fusion = f32[128,128]{1,0} fusion(lhs, rhs),
      kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__triton_gemm"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();

  CodegenBackend* direct = nullptr;
  CodegenBackend* fission = nullptr;
  for (const std::unique_ptr<CodegenBackend>& backend : backends) {
    if (backend->backend() == Backend::FLY) direct = backend.get();
    if (backend->backend() == Backend::FLY_FISSION) fission = backend.get();
  }
  ASSERT_NE(direct, nullptr);
  ASSERT_NE(fission, nullptr);
  TF_ASSERT_OK_AND_ASSIGN(auto direct_configs,
                          direct->GetSupportedConfigs(*fusion));
  TF_ASSERT_OK_AND_ASSIGN(auto fission_configs,
                          fission->GetSupportedConfigs(*fusion));
  EXPECT_TRUE(direct_configs.empty());
  EXPECT_FALSE(fission_configs.empty());
}

TEST_P(FactoryTest, NativeFlyOwnsLearnedScaleGemm) {
  if (GetParam().names !=
      std::vector<Backend>{Backend::FLY, Backend::FLY_FISSION,
                           Backend::FLY_FUSION}) {
    GTEST_SKIP() << "Only applies to the ROCm Fly backend set.";
  }
  if (!stream_executor_->GetDeviceDescription()
           .gpu_compute_capability()
           .IsRocm()) {
    GTEST_SKIP() << "Fly is a ROCm backend.";
  }

  debug_options_.set_xla_gpu_enable_flydsl_gemm(true);
  auto& registry =
      stream_executor::PlatformObjectRegistry::GetGlobalRegistry();
  TF_ASSERT_OK_AND_ASSIGN(
      const GetCodegenBackends::Type& get_codegen_backends,
      registry.FindObject<GetCodegenBackends>(platform_->id()));
  mlir::MLIRContext mlir_context;
  AliasInfo alias_info;
  xla::RegisterSymbolicExprStorage(&mlir_context);
  std::vector<std::unique_ptr<CodegenBackend>> backends =
      get_codegen_backends(
          stream_executor_, &allocator_, &debug_options_, compiler_.get(),
          &target_config_, &alias_info, &mlir_context,
          /*shape_size_fn=*/[](const Shape&) { return 0; }, GetParam().names);

  constexpr char kHlo[] = R"(
HloModule fly_learned_scale_ownership

gemm {
  data = bf16[128,256]{1,0} parameter(0)
  data_f32 = f32[128,256]{1,0} convert(data)
  scale = bf16[256]{0} parameter(1)
  scale_broadcast = bf16[128,256]{1,0} broadcast(scale), dimensions={1}
  scale_f32 = f32[128,256]{1,0} convert(scale_broadcast)
  scaled_f32 = f32[128,256]{1,0} multiply(data_f32, scale_f32)
  scaled = bf16[128,256]{1,0} convert(scaled_f32)
  weight = bf16[256,768]{1,0} parameter(2)
  ROOT dot = bf16[128,768]{1,0} dot(scaled, weight),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY main {
  data = bf16[128,256]{1,0} parameter(0)
  scale = bf16[256]{0} parameter(1)
  weight = bf16[256,768]{1,0} parameter(2)
  ROOT fusion = bf16[128,768]{1,0} fusion(data, scale, weight),
      kind=kCustom, calls=gemm,
      backend_config={"fusion_backend_config":{"kind":"__fly_gemm"}}
})";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();

  CodegenBackend* direct = nullptr;
  CodegenBackend* fission = nullptr;
  for (const std::unique_ptr<CodegenBackend>& backend : backends) {
    if (backend->backend() == Backend::FLY) direct = backend.get();
    if (backend->backend() == Backend::FLY_FISSION) fission = backend.get();
  }
  ASSERT_NE(direct, nullptr);
  ASSERT_NE(fission, nullptr);
  TF_ASSERT_OK_AND_ASSIGN(auto direct_configs,
                          direct->GetSupportedConfigs(*fusion));
  TF_ASSERT_OK_AND_ASSIGN(auto fission_configs,
                          fission->GetSupportedConfigs(*fusion));
  EXPECT_EQ(direct_configs.size(), 1);
  EXPECT_TRUE(fission_configs.empty());
}

INSTANTIATE_TEST_SUITE_P(
    All, FactoryTest,
    ::testing::Values(
        FactoryTestParams{{}, 6, /*run_on_cuda=*/true, /*run_on_rocm=*/false},
        FactoryTestParams{{}, 9, /*run_on_cuda=*/false, /*run_on_rocm=*/true},
        FactoryTestParams{{Backend::TRITON}, 1},
        FactoryTestParams{{Backend::TRITON, Backend::CUBLASLT},
                          2,
                          /*run_on_cuda=*/true,
                          /*run_on_rocm=*/false},
        FactoryTestParams{{Backend::TRITON, Backend::HIPBLASLT},
                          2,
                          /*run_on_cuda=*/false,
                          /*run_on_rocm=*/true},
        FactoryTestParams{
            {Backend::FLY, Backend::FLY_FISSION, Backend::FLY_FUSION},
            3,
            /*run_on_cuda=*/false,
            /*run_on_rocm=*/true}));

}  // namespace
}  // namespace gpu
}  // namespace xla
