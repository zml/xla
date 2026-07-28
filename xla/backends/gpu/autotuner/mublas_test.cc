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

#include "xla/backends/gpu/autotuner/mublas.h"

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

constexpr char kMublasCustomCallHlo[] = R"hlo(
HloModule mublas_autotune

ENTRY main {
  lhs = f32[8,16]{1,0} parameter(0)
  rhs = f32[16,4]{1,0} parameter(1)
  ROOT gemm = f32[8,4]{1,0} custom-call(lhs, rhs),
    custom_call_target="__mublas$gemm",
    backend_config={"gemm_backend_config":{
      "alpha_real":1,
      "beta":0,
      "dot_dimension_numbers":{
        "lhs_contracting_dimensions":["1"],
        "rhs_contracting_dimensions":["0"]
      },
      "precision_config":{"algorithm":"ALG_UNSET"},
      "epilogue":"DEFAULT"
    }}
}
)hlo";

constexpr char kUnsupportedHlo[] = R"hlo(
HloModule unsupported

ENTRY main {
  lhs = f32[8,16]{1,0} parameter(0)
  rhs = f32[16,4]{1,0} parameter(1)
  ROOT dot = f32[8,4]{1,0} dot(lhs, rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)hlo";

constexpr char kMublasF64CustomCallHlo[] = R"hlo(
HloModule mublas_f64_autotune

ENTRY main {
  lhs = f64[8,16]{1,0} parameter(0)
  rhs = f64[16,4]{1,0} parameter(1)
  ROOT gemm = f64[8,4]{1,0} custom-call(lhs, rhs),
    custom_call_target="__mublas$gemm",
    backend_config={"gemm_backend_config":{
      "alpha_real":1,
      "beta":0,
      "dot_dimension_numbers":{
        "lhs_contracting_dimensions":["1"],
        "rhs_contracting_dimensions":["0"]
      },
      "precision_config":{"algorithm":"ALG_UNSET"},
      "epilogue":"DEFAULT"
    }}
}
)hlo";

constexpr char kF64MublasCustomCallHlo[] = R"hlo(
HloModule mublas_f64_autotune

ENTRY main {
  lhs = f64[8,16]{1,0} parameter(0)
  rhs = f64[16,4]{1,0} parameter(1)
  ROOT gemm = f64[8,4]{1,0} custom-call(lhs, rhs),
    custom_call_target="__mublas$gemm",
    backend_config={"gemm_backend_config":{
      "alpha_real":1,
      "beta":0,
      "dot_dimension_numbers":{
        "lhs_contracting_dimensions":["1"],
        "rhs_contracting_dimensions":["0"]
      }
    }}
}
)hlo";

class MublasBackendTest : public HloHardwareIndependentTestBase {
 protected:
  void SetUp() override {
    HloHardwareIndependentTestBase::SetUp();
    ASSERT_OK_AND_ASSIGN(stream_executor::GpuTargetConfigProto proto,
                         GetGpuTargetConfig(GpuModel::S80));
    ASSERT_OK_AND_ASSIGN(Compiler::GpuTargetConfig target_config,
                         Compiler::GpuTargetConfig::FromProto(proto));
    target_config_ =
        std::make_unique<Compiler::GpuTargetConfig>(std::move(target_config));
    backend_ = std::make_unique<MublasBackend>(
        /*stream_executor=*/nullptr, &debug_options_, /*compiler=*/nullptr,
        target_config_.get());
  }

  DebugOptions debug_options_;
  std::unique_ptr<Compiler::GpuTargetConfig> target_config_;
  std::unique_ptr<MublasBackend> backend_;
};

TEST_F(MublasBackendTest, ReturnsNormalizedZeroWorkspaceAlgorithms) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kMublasCustomCallHlo));
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_->GetSupportedConfigs(
                           *module->entry_computation()->root_instruction()));
  ASSERT_EQ(configs.size(), 2);
  EXPECT_THAT(std::vector<int64_t>({configs[0]->gemm().algorithm(),
                                    configs[1]->gemm().algorithm()}),
              ElementsAre(kMublasDefaultAlgorithm, kMublasTensorOpAlgorithm));
  EXPECT_EQ(configs[0]->gemm().autotune_workspace_size(), 0);
  EXPECT_EQ(configs[1]->gemm().autotune_workspace_size(), 0);
}

TEST_F(MublasBackendTest, TensorOpAlgorithmIsLimitedToHomogeneousF32) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kF64MublasCustomCallHlo));
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_->GetSupportedConfigs(
                           *module->entry_computation()->root_instruction()));
  ASSERT_EQ(configs.size(), 1);
  EXPECT_EQ(configs[0]->gemm().algorithm(), kMublasDefaultAlgorithm);
}

TEST_F(MublasBackendTest, ReturnsNoConfigForUnsupportedInstruction) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kUnsupportedHlo));
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_->GetSupportedConfigs(
                           *module->entry_computation()->root_instruction()));
  EXPECT_TRUE(configs.empty());
}

TEST_F(MublasBackendTest, DefaultConfigIsAlgorithmZeroWithoutWorkspace) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kMublasCustomCallHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                       backend_->GetDefaultConfig(
                           *module->entry_computation()->root_instruction()));
  ASSERT_TRUE(config->has_gemm());
  EXPECT_EQ(config->gemm().algorithm(), kMublasDefaultAlgorithm);
  EXPECT_EQ(config->gemm().autotune_workspace_size(), 0);
}

TEST_F(MublasBackendTest, DeterminismFiltersUnqualifiedAlternative) {
  debug_options_.set_xla_gpu_deterministic_ops(true);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kMublasCustomCallHlo));
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_->GetSupportedConfigs(
                           *module->entry_computation()->root_instruction()));
  ASSERT_EQ(configs.size(), 1);
  EXPECT_EQ(configs[0]->gemm().algorithm(), kMublasDefaultAlgorithm);
}

TEST_F(MublasBackendTest, ApplyConfigWritesSelectedAlgorithm) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kMublasCustomCallHlo));
  BackendConfig config;
  config.mutable_gemm()->set_algorithm(kMublasTensorOpAlgorithm);
  config.mutable_gemm()->set_autotune_workspace_size(0);
  HloInstruction* gemm = module->entry_computation()->root_instruction();
  ASSERT_OK(backend_->ApplyConfig(*gemm, config));
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_config,
                       gemm->backend_config<GpuBackendConfig>());
  EXPECT_EQ(gpu_config.gemm_backend_config().selected_algorithm(),
            kMublasTensorOpAlgorithm);
  EXPECT_EQ(gpu_config.gemm_backend_config().autotune_workspace_size(), 0);
}

TEST_F(MublasBackendTest, RejectsUnknownAlgorithmAndWorkspace) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kMublasCustomCallHlo));
  HloInstruction* gemm = module->entry_computation()->root_instruction();

  BackendConfig unknown_algorithm;
  unknown_algorithm.mutable_gemm()->set_algorithm(2);
  EXPECT_THAT(backend_->ApplyConfig(*gemm, unknown_algorithm),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("expected 0 or 1")));

  BackendConfig workspace;
  workspace.mutable_gemm()->set_algorithm(kMublasDefaultAlgorithm);
  workspace.mutable_gemm()->set_autotune_workspace_size(1);
  EXPECT_THAT(backend_->ApplyConfig(*gemm, workspace),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("zero workspace")));
}

TEST_F(MublasBackendTest, RejectsTensorOpForF64OrDeterministicCacheEntry) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> f64_module,
                       ParseAndReturnVerifiedModule(kMublasF64CustomCallHlo));
  BackendConfig tensor_op;
  tensor_op.mutable_gemm()->set_algorithm(kMublasTensorOpAlgorithm);
  tensor_op.mutable_gemm()->set_autotune_workspace_size(0);
  EXPECT_THAT(
      backend_->ApplyConfig(
          *f64_module->entry_computation()->root_instruction(), tensor_op),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("only for homogeneous f32")));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> f32_module,
                       ParseAndReturnVerifiedModule(kMublasCustomCallHlo));
  debug_options_.set_xla_gpu_deterministic_ops(true);
  EXPECT_THAT(
      backend_->ApplyConfig(
          *f32_module->entry_computation()->root_instruction(), tensor_op),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("requires normalized algorithm 0")));
}

TEST_F(MublasBackendTest, VersionContainsTargetAndExplicitLibraryState) {
  const std::string version = backend_->version();
  EXPECT_THAT(version, HasSubstr("musa_arch=mp_21"));
  EXPECT_THAT(version, HasSubstr("musa_runtime=1.5.4"));
  EXPECT_THAT(version, HasSubstr("musa_kernel_driver=3.0.0"));
  EXPECT_THAT(version, HasSubstr("musa_toolkit=4.0.1"));
  EXPECT_THAT(version,
              HasSubstr("mublas_required_advanced_contract=xla-musa-mublas;"
                        "abi=2;base=7;advanced=63;workspace=0"));
  EXPECT_THAT(version, HasSubstr("mublas_deterministic_ops=0"));
  EXPECT_THAT(version, HasSubstr("mublas_exclude_nondeterministic_ops=0"));
  EXPECT_THAT(version, HasSubstr("mublas_shim_abi=deviceless"));
  EXPECT_THAT(version, HasSubstr("mublas=unavailable"));
}

TEST_F(MublasBackendTest, VersionSeparatesDeterminismModes) {
  const std::string default_version = backend_->version();

  debug_options_.set_xla_gpu_deterministic_ops(true);
  const std::string deterministic_version = backend_->version();
  EXPECT_NE(default_version, deterministic_version);
  EXPECT_THAT(deterministic_version, HasSubstr("mublas_deterministic_ops=1"));
  EXPECT_THAT(deterministic_version,
              HasSubstr("mublas_exclude_nondeterministic_ops=0"));

  debug_options_.set_xla_gpu_deterministic_ops(false);
  debug_options_.set_xla_gpu_exclude_nondeterministic_ops(true);
  const std::string exclude_nondeterministic_version = backend_->version();
  EXPECT_NE(default_version, exclude_nondeterministic_version);
  EXPECT_NE(deterministic_version, exclude_nondeterministic_version);
  EXPECT_THAT(exclude_nondeterministic_version,
              HasSubstr("mublas_deterministic_ops=0"));
  EXPECT_THAT(exclude_nondeterministic_version,
              HasSubstr("mublas_exclude_nondeterministic_ops=1"));
}

}  // namespace
}  // namespace xla::gpu
