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

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/thunk_emitter.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/musa/musa_compute_capability.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

std::unique_ptr<HloComputation> MakeMusaGemmComputation(
    PrimitiveType lhs_type = F32, PrimitiveType rhs_type = F32,
    PrimitiveType output_type = F32, bool batched = false) {
  HloComputation::Builder builder("main");
  const Shape lhs_shape =
      ShapeUtil::MakeShape(lhs_type, batched ? std::vector<int64_t>{5, 2, 3}
                                             : std::vector<int64_t>{2, 3});
  const Shape rhs_shape =
      ShapeUtil::MakeShape(rhs_type, batched ? std::vector<int64_t>{5, 3, 4}
                                             : std::vector<int64_t>{3, 4});
  const Shape output_shape =
      ShapeUtil::MakeShape(output_type, batched ? std::vector<int64_t>{5, 2, 4}
                                                : std::vector<int64_t>{2, 4});
  HloInstruction* lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, lhs_shape, "lhs"));
  HloInstruction* rhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, rhs_shape, "rhs"));
  HloInstruction* call =
      builder.AddInstruction(HloInstruction::CreateCustomCall(
          output_shape, {lhs, rhs}, "__mublas$gemm"));
  return builder.Build(call);
}

stream_executor::GpuComputeCapability MusaCapability() {
  return stream_executor::GpuComputeCapability(
      stream_executor::MusaComputeCapability("mp_21", 2, 1,
                                             /*hardware_warp_size=*/128,
                                             /*logical_subgroup_size=*/32));
}

GemmBackendConfig BasicMusaGemmConfig() {
  GemmBackendConfig config;
  config.set_alpha_real(1.0);
  config.set_alpha_imag(0.0);
  config.set_beta(0.0);
  config.set_epilogue(GemmBackendConfig::DEFAULT);
  config.mutable_precision_config()->set_algorithm(PrecisionConfig::ALG_UNSET);
  config.mutable_dot_dimension_numbers()->add_lhs_contracting_dimensions(1);
  config.mutable_dot_dimension_numbers()->add_rhs_contracting_dimensions(0);
  return config;
}

absl::Status ValidateCustomCall(
    const HloComputation& computation,
    const stream_executor::GpuComputeCapability& gpu_compute_capability,
    const GemmBackendConfig& config = BasicMusaGemmConfig()) {
  return thunk_emitter_internal::ValidateMusaGemmCustomCall(
      *Cast<HloCustomCallInstruction>(computation.root_instruction()),
      gpu_compute_capability, config);
}

TEST(MusaGemmBackendConfigTest, AcceptsBasicConfig) {
  EXPECT_THAT(thunk_emitter_internal::ValidateMusaGemmBackendConfig(
                  BasicMusaGemmConfig()),
              IsOk());
}

TEST(MusaGemmBackendConfigTest, RejectsNonzeroBeta) {
  GemmBackendConfig config = BasicMusaGemmConfig();
  config.set_beta(1.0);
  EXPECT_THAT(thunk_emitter_internal::ValidateMusaGemmBackendConfig(config),
              StatusIs(absl::StatusCode::kUnimplemented, HasSubstr("beta=0")));
}

TEST(MusaGemmBackendConfigTest, RejectsNonDefaultEpilogue) {
  GemmBackendConfig config = BasicMusaGemmConfig();
  config.set_epilogue(GemmBackendConfig::RELU);
  EXPECT_THAT(
      thunk_emitter_internal::ValidateMusaGemmBackendConfig(config),
      StatusIs(absl::StatusCode::kUnimplemented, HasSubstr("epilogue")));
}

TEST(MusaGemmBackendConfigTest, AcceptsQualifiedSelectedAlgorithms) {
  GemmBackendConfig config = BasicMusaGemmConfig();
  config.set_selected_algorithm(0);
  EXPECT_THAT(thunk_emitter_internal::ValidateMusaGemmBackendConfig(config),
              IsOk());
  config.set_selected_algorithm(1);
  EXPECT_THAT(thunk_emitter_internal::ValidateMusaGemmBackendConfig(config),
              IsOk());
}

TEST(MusaGemmBackendConfigTest, RejectsUnknownSelectedAlgorithm) {
  GemmBackendConfig config = BasicMusaGemmConfig();
  config.set_selected_algorithm(7);
  EXPECT_THAT(thunk_emitter_internal::ValidateMusaGemmBackendConfig(config),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must be 0 (default) or 1")));
}

TEST(MusaGemmBackendConfigTest, RejectsNonDefaultPrecisionAlgorithm) {
  GemmBackendConfig config = BasicMusaGemmConfig();
  config.mutable_precision_config()->set_algorithm(
      PrecisionConfig::ALG_DOT_F32_F32_F32);
  EXPECT_THAT(
      thunk_emitter_internal::ValidateMusaGemmBackendConfig(config),
      StatusIs(absl::StatusCode::kUnimplemented, HasSubstr("ALG_UNSET")));
}

TEST(MusaGemmBackendConfigTest, RejectsScaleMode) {
  GemmBackendConfig config = BasicMusaGemmConfig();
  config.set_scale_mode(1);
  EXPECT_THAT(thunk_emitter_internal::ValidateMusaGemmBackendConfig(config),
              StatusIs(absl::StatusCode::kUnimplemented, HasSubstr("scaling")));
}

TEST(MusaGemmCustomCallTest, AcceptsHomogeneousF32AndF64OnMusa) {
  std::unique_ptr<HloComputation> f32 = MakeMusaGemmComputation();
  std::unique_ptr<HloComputation> f64 = MakeMusaGemmComputation(F64, F64, F64);
  EXPECT_THAT(ValidateCustomCall(*f32, MusaCapability()), IsOk());
  EXPECT_THAT(ValidateCustomCall(*f64, MusaCapability()), IsOk());
}

TEST(MusaGemmCustomCallTest, RejectsCudaCapability) {
  std::unique_ptr<HloComputation> computation = MakeMusaGemmComputation();
  EXPECT_THAT(
      ValidateCustomCall(*computation,
                         stream_executor::GpuComputeCapability(
                             stream_executor::CudaComputeCapability(9, 0))),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("requires a MUSA compute capability")));
}

TEST(MusaGemmCustomCallTest, RejectsRocmCapability) {
  std::unique_ptr<HloComputation> computation = MakeMusaGemmComputation();
  EXPECT_THAT(
      ValidateCustomCall(*computation,
                         stream_executor::GpuComputeCapability(
                             stream_executor::RocmComputeCapability("gfx942"))),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("requires a MUSA compute capability")));
}

TEST(MusaGemmCustomCallTest, RejectsLowPrecisionAndMixedSchemas) {
  std::unique_ptr<HloComputation> f16 = MakeMusaGemmComputation(F16, F16, F16);
  std::unique_ptr<HloComputation> bf16 =
      MakeMusaGemmComputation(BF16, BF16, BF16);
  std::unique_ptr<HloComputation> mixed =
      MakeMusaGemmComputation(F32, F64, F32);
  EXPECT_THAT(ValidateCustomCall(*f16, MusaCapability()),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("homogeneous F32 or F64")));
  EXPECT_THAT(ValidateCustomCall(*bf16, MusaCapability()),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("homogeneous F32 or F64")));
  EXPECT_THAT(ValidateCustomCall(*mixed, MusaCapability()),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("homogeneous F32 or F64")));
}

TEST(MusaGemmCustomCallTest, RejectsOutputToOperandAliasing) {
  std::unique_ptr<HloComputation> computation = MakeMusaGemmComputation();
  Cast<HloCustomCallInstruction>(computation->root_instruction())
      ->set_output_to_operand_aliasing({{{}, {0, {}}}});
  EXPECT_THAT(ValidateCustomCall(*computation, MusaCapability()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("output-to-operand aliasing")));
}

TEST(MusaGemmCustomCallTest, AcceptsMatchedBatchedSchema) {
  std::unique_ptr<HloComputation> computation =
      MakeMusaGemmComputation(F32, F32, F32, /*batched=*/true);
  GemmBackendConfig config = BasicMusaGemmConfig();
  config.mutable_dot_dimension_numbers()->set_lhs_contracting_dimensions(0, 2);
  config.mutable_dot_dimension_numbers()->set_rhs_contracting_dimensions(0, 1);
  config.mutable_dot_dimension_numbers()->add_lhs_batch_dimensions(0);
  config.mutable_dot_dimension_numbers()->add_rhs_batch_dimensions(0);
  EXPECT_THAT(ValidateCustomCall(*computation, MusaCapability(), config),
              IsOk());
}

TEST(MusaGemmCustomCallTest, RejectsMismatchedBatchSizes) {
  std::unique_ptr<HloComputation> computation =
      MakeMusaGemmComputation(F32, F32, F32, /*batched=*/true);
  computation->root_instruction()
      ->mutable_operand(1)
      ->mutable_shape()
      ->set_dimensions(0, 7);
  GemmBackendConfig config = BasicMusaGemmConfig();
  config.mutable_dot_dimension_numbers()->set_lhs_contracting_dimensions(0, 2);
  config.mutable_dot_dimension_numbers()->set_rhs_contracting_dimensions(0, 1);
  config.mutable_dot_dimension_numbers()->add_lhs_batch_dimensions(0);
  config.mutable_dot_dimension_numbers()->add_rhs_batch_dimensions(0);
  EXPECT_THAT(ValidateCustomCall(*computation, MusaCapability(), config),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("batch dimensions must have equal sizes")));
}

TEST(MusaAdvancedEmitterPolicyTest, RejectsTritonBeforeLowering) {
  EXPECT_THAT(
      thunk_emitter_internal::ValidateTritonCustomCallPlatform(
          MusaCapability()),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("not qualified for the MUSA backend")));
}

TEST(MusaAdvancedEmitterPolicyTest, PreservesOtherGpuBackends) {
  stream_executor::GpuComputeCapability cuda(
      stream_executor::CudaComputeCapability(9, 0));
  EXPECT_THAT(
      thunk_emitter_internal::ValidateTritonCustomCallPlatform(cuda), IsOk());
}

}  // namespace
}  // namespace xla::gpu
