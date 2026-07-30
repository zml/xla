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

#include "xla/service/gpu/musa_gpu_compiler.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/service/compilation_stats.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/musa/musa_compilation_provider.h"
#include "xla/service/gpu/musa/musa_llvm14_compatibility.h"
#include "xla/service/gpu/musa/musa_shim_abi.h"
#include "xla/service/gpu/target_constants.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/musa/musa_compute_capability.h"
#include "xla/stream_executor/musa/musa_mublas_api.h"
#include "xla/stream_executor/musa/musa_mudnn_api.h"
#include "xla/stream_executor/musa/musa_mufft_api.h"
#include "xla/stream_executor/musa/musa_optional_library_abi.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::IsEmpty;

class RecordingCompilationProvider final
    : public musa::MusaCompilationProvider {
 public:
  absl::StatusOr<musa::MusaCompilationArtifact> Compile(
      const musa::MusaLlvm14CompatibilityResult& module,
      const musa::MusaCompilationOptions& options) const override {
    ++compile_count;
    last_module = module;
    last_options = options;
    return musa::MusaCompilationArtifact{
        .mubin = {0x7f, 'E', 'L', 'F'},
        .mubin_sha256 = std::string(64, '0'),
        .cache_key = std::string(64, '1'),
    };
  }

  const musa::MusaCompilationIdentity& identity() const override {
    return identity_;
  }
  musa::MusaCompilationCapabilities capabilities() const override { return {}; }
  absl::string_view name() const override { return "recording"; }

  mutable int compile_count = 0;
  mutable std::optional<musa::MusaLlvm14CompatibilityResult> last_module;
  mutable std::optional<musa::MusaCompilationOptions> last_options;

 private:
  musa::MusaCompilationIdentity identity_{
      .xla_revision = "xla-test-revision",
      .current_llvm_revision = "llvm-test-revision",
      .provider_name = "recording",
      .provider_fingerprint = std::string(64, '1'),
      .bridge_fingerprint = std::string(64, '2'),
      .toolchain_fingerprint = std::string(64, '3'),
      .libdevice_fingerprint = std::string(64, '4'),
      .driver_compatibility = "musa-driver-3.0-compatible",
      .runtime_compatibility = "musa-runtime-4.0.1-compatible",
  };
};

class TestMusaGpuCompiler : public MusaGpuCompiler {
 public:
  using MusaGpuCompiler::CompileTargetBinary;
  using MusaGpuCompiler::CreateExecutableAbiVersion;
  using MusaGpuCompiler::MusaGpuCompiler;
  using MusaGpuCompiler::OptimizeHloPostLayoutAssignment;
  using MusaGpuCompiler::UseAotCompiledThunks;
  using MusaGpuCompiler::ValidatePersistentKernelCache;
};

std::unique_ptr<llvm::Module> ElementalModule(llvm::LLVMContext& context) {
  const std::string source = absl::StrCat(
      "source_filename = \"c10-test\"\n", "target datalayout = \"",
      musa::DataLayout(), "\"\n", "target triple = \"", musa::TargetTriple(),
      "\"\n\n", "define void @kernel(ptr addrspace(1) %out) #0 {\n", "entry:\n",
      "  store i32 7, ptr addrspace(1) %out, align 4\n", "  ret void\n",
      "}\n\n", "attributes #0 = { \"", musa::kMusaLlvmKernelMarker, "\" }\n");
  llvm::SMDiagnostic diagnostic;
  return llvm::parseAssemblyString(source, diagnostic, context);
}

stream_executor::DeviceDescription MusaDevice() {
  stream_executor::DeviceDescription device;
  device.set_threads_per_warp(128);
  device.set_gpu_compute_capability(stream_executor::GpuComputeCapability(
      stream_executor::MusaComputeCapability("mp_21", 2, 1,
                                             /*hardware_warp_size=*/128,
                                             /*logical_subgroup_size=*/32)));
  device.set_device_address_bits(64);
  device.set_runtime_version(stream_executor::SemanticVersion(1, 5, 4));
  device.set_driver_version(stream_executor::SemanticVersion(1, 5, 4));
  device.set_kernel_mode_driver_version(
      stream_executor::SemanticVersion(3, 0, 0));
  device.set_compile_time_toolkit_version(
      stream_executor::SemanticVersion(4, 0, 1));
  return device;
}

constexpr absl::string_view kForwardConvolutionHlo = R"(
HloModule musa_mudnn_forward

ENTRY main {
  input = f32[8,128,2,32]{3,2,1,0} parameter(0)
  filter = f32[3,3,128,128]{3,2,1,0} parameter(1)
  ROOT conv = f32[8,128,2,32]{3,2,1,0} convolution(input, filter),
    window={size=3x3 pad=1_1x1_1}, dim_labels=bf01_01io->bf01
}
)";

constexpr absl::string_view kBackwardInputConvolutionHlo = R"(
HloModule musa_mudnn_backward_input

ENTRY main {
  output = f32[4,5,16,16]{3,2,1,0} parameter(0)
  kernel = f32[5,3,7,7]{3,2,1,0} parameter(1)
  reverse = f32[5,3,7,7]{3,2,1,0} reverse(kernel), dimensions={2,3}
  ROOT conv = f32[4,3,16,16]{3,2,1,0} convolution(output, reverse),
    window={size=7x7 pad=3_3x3_3}, dim_labels=bf01_io01->bf01
}
)";

constexpr absl::string_view kBackwardFilterConvolutionHlo = R"(
HloModule musa_mudnn_backward_filter

ENTRY main {
  activations = f32[1,1,3,1]{3,2,1,0} parameter(0)
  gradients = f32[1,1,2,1]{3,2,1,0} parameter(1)
  ROOT conv = f32[1,1,1,1]{3,2,1,0}
    convolution(activations, gradients),
    window={size=1x2 rhs_dilate=1x2}, dim_labels=f01b_i01o->01bf
}
)";

const HloInstruction* FindDnnConvolutionCustomCall(const HloModule& module) {
  for (const HloComputation* computation : module.computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      if (IsCustomCallToDnnConvolution(*instruction)) return instruction;
    }
  }
  return nullptr;
}

TEST(MusaGpuCompilerTest, BuildsFullEnvelopeAndForcesCompiledThunkAot) {
  TestMusaGpuCompiler compiler(
      std::make_unique<RecordingCompilationProvider>());
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(R"(
HloModule envelope

ENTRY main {
  ROOT result = f32[] constant(1)
}
)"));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_experimental_aot_compiled_thunks(false);

  ASSERT_OK_AND_ASSIGN(
      stream_executor::ExecutableAbiVersion version,
      compiler.CreateExecutableAbiVersion(*module, MusaDevice(), {}));
  ASSERT_TRUE(version.proto().has_musa_platform_version());
  EXPECT_EQ(version.proto().musa_platform_version().architecture(), "mp_21");
  EXPECT_EQ(version.proto().musa_platform_version().binary_kind(), "mubin");
  EXPECT_EQ(version.proto()
                .musa_platform_version()
                .required_optional_library_abis_size(),
            0);
  EXPECT_TRUE(compiler.UseAotCompiledThunks(*module));
}

TEST(MusaGpuCompilerTest, MissingMudnnLeavesConvolutionOnGenericPath) {
  TestMusaGpuCompiler compiler(
      std::make_unique<RecordingCompilationProvider>());
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnUnverifiedModule(kForwardConvolutionHlo));
  std::unique_ptr<CompilationStats> stats = CompilationStats::MakeNoopStats();
  const stream_executor::DeviceDescription device = MusaDevice();

  EXPECT_THAT(compiler.OptimizeHloConvolutionCanonicalization(
                  module.get(), device.gpu_compute_capability(),
                  stream_executor::dnn::VersionInfo(),
                  stream_executor::SemanticVersion(4, 0, 1),
                  /*is_deviceless=*/false, stats.get()),
              IsOk());
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kConvolution);
  EXPECT_EQ(FindDnnConvolutionCustomCall(*module), nullptr);
}

TEST(MusaGpuCompilerTest, QualifiedMudnnRewritesToSharedDnnCustomCall) {
  TestMusaGpuCompiler compiler(
      std::make_unique<RecordingCompilationProvider>());
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnUnverifiedModule(kForwardConvolutionHlo));
  std::unique_ptr<CompilationStats> stats = CompilationStats::MakeNoopStats();
  const stream_executor::DeviceDescription device = MusaDevice();

  EXPECT_THAT(compiler.OptimizeHloConvolutionCanonicalization(
                  module.get(), device.gpu_compute_capability(),
                  stream_executor::dnn::VersionInfo(2, 8, 0),
                  stream_executor::SemanticVersion(4, 0, 1),
                  /*is_deviceless=*/false, stats.get()),
              IsOk());
  const HloInstruction* custom_call = FindDnnConvolutionCustomCall(*module);
  ASSERT_NE(custom_call, nullptr);
  EXPECT_EQ(custom_call->custom_call_target(), kCudnnConvForwardCallTarget);
  ASSERT_TRUE(custom_call->shape().IsTuple());
  EXPECT_EQ(custom_call->shape().tuple_shapes(1).dimensions(0), 0);
}

TEST(MusaGpuCompilerTest, QualifiedAlignedF16MudnnRewritesToSharedDnnCall) {
  TestMusaGpuCompiler compiler(
      std::make_unique<RecordingCompilationProvider>());
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(R"(
HloModule musa_mudnn_f16_aligned
ENTRY main {
  input = f16[1,4,4,8]{3,2,1,0} parameter(0)
  filter = f16[3,3,8,8]{3,2,1,0} parameter(1)
  ROOT conv = f16[1,4,4,8]{3,2,1,0} convolution(input, filter),
    window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
}
)"));
  std::unique_ptr<CompilationStats> stats = CompilationStats::MakeNoopStats();
  const stream_executor::DeviceDescription device = MusaDevice();

  ASSERT_OK(compiler.OptimizeHloConvolutionCanonicalization(
      module.get(), device.gpu_compute_capability(),
      stream_executor::dnn::VersionInfo(2, 8, 0),
      stream_executor::SemanticVersion(4, 0, 1),
      /*is_deviceless=*/false, stats.get()));
  const HloInstruction* custom_call = FindDnnConvolutionCustomCall(*module);
  ASSERT_NE(custom_call, nullptr);
  EXPECT_EQ(custom_call->custom_call_target(), kCudnnConvForwardCallTarget);
}

TEST(MusaGpuCompilerTest, QualifiedMudnnRewritesBackwardDataAndFilter) {
  TestMusaGpuCompiler compiler(
      std::make_unique<RecordingCompilationProvider>());
  const stream_executor::DeviceDescription device = MusaDevice();

  for (const auto& [hlo, target] :
       std::vector<std::pair<absl::string_view, absl::string_view>>{
           {kBackwardInputConvolutionHlo, kCudnnConvBackwardInputCallTarget},
           {kBackwardFilterConvolutionHlo,
            kCudnnConvBackwardFilterCallTarget}}) {
    ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
    std::unique_ptr<CompilationStats> stats = CompilationStats::MakeNoopStats();
    ASSERT_OK(compiler.OptimizeHloConvolutionCanonicalization(
        module.get(), device.gpu_compute_capability(),
        stream_executor::dnn::VersionInfo(2, 8, 0),
        stream_executor::SemanticVersion(4, 0, 1),
        /*is_deviceless=*/false, stats.get()));
    const HloInstruction* custom_call = FindDnnConvolutionCustomCall(*module);
    ASSERT_NE(custom_call, nullptr);
    EXPECT_EQ(custom_call->custom_call_target(), target);
  }
}

TEST(MusaGpuCompilerTest, DevicelessCompilationKeepsGenericConvolution) {
  TestMusaGpuCompiler compiler(
      std::make_unique<RecordingCompilationProvider>());
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnUnverifiedModule(kForwardConvolutionHlo));
  std::unique_ptr<CompilationStats> stats = CompilationStats::MakeNoopStats();
  const stream_executor::DeviceDescription device = MusaDevice();

  ASSERT_OK(compiler.OptimizeHloConvolutionCanonicalization(
      module.get(), device.gpu_compute_capability(),
      stream_executor::dnn::VersionInfo(2, 8, 0),
      stream_executor::SemanticVersion(4, 0, 1),
      /*is_deviceless=*/true, stats.get()));
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kConvolution);
  EXPECT_EQ(FindDnnConvolutionCustomCall(*module), nullptr);
}

TEST(MusaGpuCompilerTest, UnsupportedMudnnContractsStayGeneric) {
  constexpr absl::string_view kUnsupportedHlos[] = {
      R"(
HloModule musa_mudnn_f64_generic
ENTRY main {
  input = f64[2,4,4,3]{3,2,1,0} parameter(0)
  filter = f64[3,3,3,5]{3,2,1,0} parameter(1)
  ROOT conv = f64[2,4,4,5]{3,2,1,0} convolution(input, filter),
    window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
}
)",
      R"(
HloModule musa_mudnn_conv1d_generic
ENTRY main {
  input = f32[1,2,1]{2,1,0} parameter(0)
  filter = f32[1,1,1]{2,1,0} parameter(1)
  ROOT conv = f32[1,2,1]{2,1,0} convolution(input, filter),
    window={size=1}, dim_labels=b0f_0io->b0f
}
)",
      R"(
HloModule musa_mudnn_reversed_generic
ENTRY main {
  input = f32[2,4,4,3]{3,2,1,0} parameter(0)
  filter = f32[3,3,3,5]{3,2,1,0} parameter(1)
  ROOT conv = f32[2,4,4,5]{3,2,1,0} convolution(input, filter),
    window={size=3x3 pad=1_1x1_1 rhs_reversal=1x1},
    dim_labels=b01f_01io->b01f
}
)",
      R"(
HloModule musa_mudnn_fp8_generic
ENTRY main {
  input = f8e4m3fn[2,4,4,3]{3,2,1,0} parameter(0)
  filter = f8e4m3fn[3,3,3,5]{3,2,1,0} parameter(1)
  ROOT conv = f8e4m3fn[2,4,4,5]{3,2,1,0} convolution(input, filter),
    window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
}
)",
      R"(
HloModule musa_mudnn_complex_generic
ENTRY main {
  input = c64[2,4,4,3]{3,2,1,0} parameter(0)
  filter = c64[3,3,3,5]{3,2,1,0} parameter(1)
  ROOT conv = c64[2,4,4,5]{3,2,1,0} convolution(input, filter),
    window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
}
)",
      R"(
HloModule musa_mudnn_bf16_generic
ENTRY main {
  input = bf16[1,4,4,8]{3,2,1,0} parameter(0)
  filter = bf16[3,3,8,8]{3,2,1,0} parameter(1)
  ROOT conv = bf16[1,4,4,8]{3,2,1,0} convolution(input, filter),
    window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
}
)",
      R"(
HloModule musa_mudnn_f16_unaligned_generic
ENTRY main {
  input = f16[1,4,4,3]{3,2,1,0} parameter(0)
  filter = f16[3,3,3,5]{3,2,1,0} parameter(1)
  ROOT conv = f16[1,4,4,5]{3,2,1,0} convolution(input, filter),
    window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
}
)"};
  const stream_executor::DeviceDescription device = MusaDevice();
  for (absl::string_view hlo : kUnsupportedHlos) {
    TestMusaGpuCompiler compiler(
        std::make_unique<RecordingCompilationProvider>());
    ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
    std::unique_ptr<CompilationStats> stats = CompilationStats::MakeNoopStats();
    ASSERT_OK(compiler.OptimizeHloConvolutionCanonicalization(
        module.get(), device.gpu_compute_capability(),
        stream_executor::dnn::VersionInfo(2, 8, 0),
        stream_executor::SemanticVersion(4, 0, 1),
        /*is_deviceless=*/false, stats.get()));
    EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
              HloOpcode::kConvolution);
    EXPECT_EQ(FindDnnConvolutionCustomCall(*module), nullptr);
  }
}

TEST(MusaGpuCompilerTest, DisabledBinaryLibrariesKeepGenericConvolution) {
  TestMusaGpuCompiler compiler(
      std::make_unique<RecordingCompilationProvider>());
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnUnverifiedModule(kForwardConvolutionHlo));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_experimental_disable_binary_libraries(true);
  std::unique_ptr<CompilationStats> stats = CompilationStats::MakeNoopStats();
  const stream_executor::DeviceDescription device = MusaDevice();

  EXPECT_THAT(compiler.OptimizeHloConvolutionCanonicalization(
                  module.get(), device.gpu_compute_capability(),
                  stream_executor::dnn::VersionInfo(2, 8, 0),
                  stream_executor::SemanticVersion(4, 0, 1),
                  /*is_deviceless=*/false, stats.get()),
              IsOk());
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kConvolution);
  EXPECT_EQ(FindDnnConvolutionCustomCall(*module), nullptr);
}

TEST(MusaGpuCompilerTest, UnqualifiedMudnnVersionKeepsGenericConvolution) {
  TestMusaGpuCompiler compiler(
      std::make_unique<RecordingCompilationProvider>());
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnUnverifiedModule(kForwardConvolutionHlo));
  std::unique_ptr<CompilationStats> stats = CompilationStats::MakeNoopStats();
  const stream_executor::DeviceDescription device = MusaDevice();

  EXPECT_THAT(compiler.OptimizeHloConvolutionCanonicalization(
                  module.get(), device.gpu_compute_capability(),
                  stream_executor::dnn::VersionInfo(2, 8, 1),
                  stream_executor::SemanticVersion(4, 0, 1),
                  /*is_deviceless=*/false, stats.get()),
              IsOk());
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kConvolution);
  EXPECT_EQ(FindDnnConvolutionCustomCall(*module), nullptr);
}

TEST(MusaGpuCompilerTest, DeterministicModeKeepsGenericConvolution) {
  TestMusaGpuCompiler compiler(
      std::make_unique<RecordingCompilationProvider>());
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnUnverifiedModule(kForwardConvolutionHlo));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_deterministic_ops(true);
  std::unique_ptr<CompilationStats> stats = CompilationStats::MakeNoopStats();
  const stream_executor::DeviceDescription device = MusaDevice();

  EXPECT_THAT(compiler.OptimizeHloConvolutionCanonicalization(
                  module.get(), device.gpu_compute_capability(),
                  stream_executor::dnn::VersionInfo(2, 8, 0),
                  stream_executor::SemanticVersion(4, 0, 1),
                  /*is_deviceless=*/false, stats.get()),
              IsOk());
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kConvolution);
  EXPECT_EQ(FindDnnConvolutionCustomCall(*module), nullptr);
}

TEST(MusaGpuCompilerTest, DnnConvolutionRequiresExactMudnnFingerprint) {
  TestMusaGpuCompiler compiler(
      std::make_unique<RecordingCompilationProvider>());
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(R"(
HloModule mudnn_envelope

ENTRY main {
  input = f32[3,56,56,16]{2,1,0,3} parameter(0)
  filter = f32[3,3,3,64]{2,1,0,3} parameter(1)
  ROOT conv = (f32[54,54,16,64]{1,0,3,2}, u8[0]{0})
    custom-call(input, filter), custom_call_target="__cudnn$convForward",
    window={size=3x3}, dim_labels=f01b_i01o->01bf,
    backend_config={"cudnn_conv_backend_config":{
      "activation_mode":"kNone",
      "conv_result_scale":1,
      "side_input_scale":0,
      "leakyrelu_alpha":0
    }}
}
)"));

  ASSERT_OK_AND_ASSIGN(
      stream_executor::ExecutableAbiVersion version,
      compiler.CreateExecutableAbiVersion(*module, MusaDevice(), {}));
  const auto& libraries =
      version.proto().musa_platform_version().required_optional_library_abis();
  ASSERT_EQ(libraries.size(), 1);
  EXPECT_EQ(libraries[0].name(),
            stream_executor::musa::kMusaMuDnnLibraryAbiName);
  EXPECT_EQ(libraries[0].abi_version(),
            stream_executor::musa::kMusaMuDnnLibraryAbiVersion);
  EXPECT_EQ(libraries[0].fingerprint(),
            stream_executor::musa::kMusaMuDnnAbiFingerprintV1);
}

TEST(MusaGpuCompilerTest, RequiresMublasOnlyForDedicatedMusaGemmTarget) {
  TestMusaGpuCompiler compiler(
      std::make_unique<RecordingCompilationProvider>());
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(R"(
HloModule mublas_envelope

ENTRY main {
  lhs = f32[8,16]{1,0} parameter(0)
  rhs = f32[16,4]{1,0} parameter(1)
  ROOT result = f32[8,4]{1,0} custom-call(lhs, rhs),
    custom_call_target="__mublas$gemm"
}
)"));

  ASSERT_OK_AND_ASSIGN(
      stream_executor::ExecutableAbiVersion version,
      compiler.CreateExecutableAbiVersion(*module, MusaDevice(), {}));
  const auto& musa = version.proto().musa_platform_version();
  ASSERT_EQ(musa.required_optional_library_abis_size(), 1);
  EXPECT_EQ(musa.required_optional_library_abis(0).name(),
            stream_executor::musa::kMusaMuBlasLibraryAbiName);
  EXPECT_EQ(musa.required_optional_library_abis(0).abi_version(),
            stream_executor::musa::kMusaMuBlasLibraryAbiVersion);
  EXPECT_TRUE(musa.required_optional_library_abis(0).fingerprint().empty());
}

TEST(MusaGpuCompilerTest, SelectedMublasAlgorithmRequiresAdvancedFingerprint) {
  TestMusaGpuCompiler compiler(
      std::make_unique<RecordingCompilationProvider>());
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(R"(
HloModule mublas_selected_algorithm_envelope

ENTRY main {
  lhs = f32[8,16]{1,0} parameter(0)
  rhs = f32[16,4]{1,0} parameter(1)
  ROOT result = f32[8,4]{1,0} custom-call(lhs, rhs),
    custom_call_target="__mublas$gemm",
    backend_config={"gemm_backend_config":{
      "selected_algorithm":"1",
      "dot_dimension_numbers":{
        "lhs_contracting_dimensions":["1"],
        "rhs_contracting_dimensions":["0"]
      }
    }}
}
)"));

  ASSERT_OK_AND_ASSIGN(
      stream_executor::ExecutableAbiVersion version,
      compiler.CreateExecutableAbiVersion(*module, MusaDevice(), {}));
  const auto& libraries =
      version.proto().musa_platform_version().required_optional_library_abis();
  ASSERT_EQ(libraries.size(), 1);
  EXPECT_EQ(libraries[0].fingerprint(),
            stream_executor::musa::kMusaMuBlasAdvancedAbiFingerprintV2);
}

TEST(MusaGpuCompilerTest, BatchedMublasRequiresAdvancedFingerprint) {
  TestMusaGpuCompiler compiler(
      std::make_unique<RecordingCompilationProvider>());
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(R"(
HloModule mublas_batched_envelope

ENTRY main {
  lhs = f32[2,8,16]{2,1,0} parameter(0)
  rhs = f32[2,16,4]{2,1,0} parameter(1)
  ROOT result = f32[2,8,4]{2,1,0} custom-call(lhs, rhs),
    custom_call_target="__mublas$gemm",
    backend_config={"gemm_backend_config":{
      "dot_dimension_numbers":{
        "lhs_contracting_dimensions":["2"],
        "rhs_contracting_dimensions":["1"],
        "lhs_batch_dimensions":["0"],
        "rhs_batch_dimensions":["0"]
      }
    }}
}
)"));

  ASSERT_OK_AND_ASSIGN(
      stream_executor::ExecutableAbiVersion version,
      compiler.CreateExecutableAbiVersion(*module, MusaDevice(), {}));
  const auto& libraries =
      version.proto().musa_platform_version().required_optional_library_abis();
  ASSERT_EQ(libraries.size(), 1);
  EXPECT_EQ(libraries[0].fingerprint(),
            stream_executor::musa::kMusaMuBlasAdvancedAbiFingerprintV2);
}

TEST(MusaGpuCompilerTest, DeterministicMublasRequiresAdvancedFingerprint) {
  TestMusaGpuCompiler compiler(
      std::make_unique<RecordingCompilationProvider>());
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(R"(
HloModule deterministic_mublas_envelope

ENTRY main {
  lhs = f32[8,16]{1,0} parameter(0)
  rhs = f32[16,4]{1,0} parameter(1)
  ROOT result = f32[8,4]{1,0} custom-call(lhs, rhs),
    custom_call_target="__mublas$gemm"
}
)"));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_deterministic_ops(true);

  ASSERT_OK_AND_ASSIGN(
      stream_executor::ExecutableAbiVersion version,
      compiler.CreateExecutableAbiVersion(*module, MusaDevice(), {}));
  const auto& libraries =
      version.proto().musa_platform_version().required_optional_library_abis();
  ASSERT_EQ(libraries.size(), 1);
  EXPECT_EQ(libraries[0].fingerprint(),
            stream_executor::musa::kMusaMuBlasAdvancedAbiFingerprintV2);
}

TEST(MusaGpuCompilerTest, ForwardFftRequiresExactMufftFingerprint) {
  TestMusaGpuCompiler compiler(
      std::make_unique<RecordingCompilationProvider>());
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(R"(
HloModule mufft_forward_envelope

ENTRY main {
  input = c64[8]{0} parameter(0)
  ROOT result = c64[8]{0} fft(input), fft_type=FFT, fft_length={8}
}
)"));

  ASSERT_OK_AND_ASSIGN(
      stream_executor::ExecutableAbiVersion version,
      compiler.CreateExecutableAbiVersion(*module, MusaDevice(), {}));
  const auto& libraries =
      version.proto().musa_platform_version().required_optional_library_abis();
  ASSERT_EQ(libraries.size(), 1);
  EXPECT_EQ(libraries[0].name(),
            stream_executor::musa::kMusaMuFftLibraryAbiName);
  EXPECT_EQ(libraries[0].abi_version(),
            stream_executor::musa::kMusaMuFftLibraryAbiVersion);
  EXPECT_EQ(libraries[0].fingerprint(),
            stream_executor::musa::kMusaMuFftAbiFingerprintV1);
}

TEST(MusaGpuCompilerTest, InverseFftRequiresMufftAndScalFingerprints) {
  TestMusaGpuCompiler compiler(
      std::make_unique<RecordingCompilationProvider>());
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(R"(
HloModule mufft_inverse_envelope

ENTRY main {
  input = c128[4,8]{1,0} parameter(0)
  ROOT result = c128[4,8]{1,0} fft(input), fft_type=IFFT,
    fft_length={4,8}
}
)"));

  ASSERT_OK_AND_ASSIGN(
      stream_executor::ExecutableAbiVersion version,
      compiler.CreateExecutableAbiVersion(*module, MusaDevice(), {}));
  const auto& libraries =
      version.proto().musa_platform_version().required_optional_library_abis();
  ASSERT_EQ(libraries.size(), 2);
  EXPECT_EQ(libraries[0].name(),
            stream_executor::musa::kMusaMuBlasScalLibraryAbiName);
  EXPECT_EQ(libraries[0].abi_version(),
            stream_executor::musa::kMusaMuBlasScalLibraryAbiVersion);
  EXPECT_EQ(libraries[0].fingerprint(),
            stream_executor::musa::kMusaMuBlasScalAbiFingerprintV1);
  EXPECT_EQ(libraries[1].name(),
            stream_executor::musa::kMusaMuFftLibraryAbiName);
  EXPECT_EQ(libraries[1].fingerprint(),
            stream_executor::musa::kMusaMuFftAbiFingerprintV1);
}

TEST(MusaGpuCompilerTest, CombinedGemmAndFftRequirementsAreCanonical) {
  TestMusaGpuCompiler compiler(
      std::make_unique<RecordingCompilationProvider>());
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(R"(
HloModule mublas_mufft_envelope

ENTRY main {
  lhs = f32[2,2]{1,0} parameter(0)
  rhs = f32[2,2]{1,0} parameter(1)
  gemm = f32[2,2]{1,0} custom-call(lhs, rhs),
    custom_call_target="__mublas$gemm"
  fft_input = c64[8]{0} parameter(2)
  transformed = c64[8]{0} fft(fft_input), fft_type=IFFT, fft_length={8}
  ROOT result = (f32[2,2]{1,0}, c64[8]{0}) tuple(gemm, transformed)
}
)"));

  ASSERT_OK_AND_ASSIGN(
      stream_executor::ExecutableAbiVersion version,
      compiler.CreateExecutableAbiVersion(*module, MusaDevice(), {}));
  const auto& libraries =
      version.proto().musa_platform_version().required_optional_library_abis();
  ASSERT_EQ(libraries.size(), 3);
  EXPECT_EQ(libraries[0].name(),
            stream_executor::musa::kMusaMuBlasLibraryAbiName);
  EXPECT_EQ(libraries[1].name(),
            stream_executor::musa::kMusaMuBlasScalLibraryAbiName);
  EXPECT_EQ(libraries[2].name(),
            stream_executor::musa::kMusaMuFftLibraryAbiName);
}

TEST(MusaGpuCompilerTest, PersistentKernelCacheFailsClosed) {
  TestMusaGpuCompiler compiler(
      std::make_unique<RecordingCompilationProvider>());
  HloModuleConfig config;
  EXPECT_THAT(compiler.ValidatePersistentKernelCache(config), IsOk());
  config.mutable_debug_options().set_xla_gpu_kernel_cache_file("cache.pb");
  EXPECT_THAT(compiler.ValidatePersistentKernelCache(config),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("complete MUSA architecture")));
}

TEST(MusaGpuCompilerTest, NormalizesAndCompilesThroughInjectedProvider) {
  auto provider = std::make_unique<RecordingCompilationProvider>();
  RecordingCompilationProvider* recording = provider.get();
  TestMusaGpuCompiler compiler(std::move(provider));
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module = ElementalModule(context);
  ASSERT_NE(module, nullptr);

  HloModuleConfig config;
  config.mutable_debug_options().set_xla_backend_optimization_level(3);
  auto result = compiler.CompileTargetBinary(
      config, module.get(), MusaDevice(), /*relocatable=*/false,
      /*debug_module=*/nullptr, /*shard_number=*/std::nullopt);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->binary, (std::vector<uint8_t>{0x7f, 'E', 'L', 'F'}));
  ASSERT_EQ(recording->compile_count, 1);
  ASSERT_TRUE(recording->last_module.has_value());
  EXPECT_EQ(recording->last_module->metadata.kernel_entry_names,
            std::vector<std::string>{"kernel"});
  ASSERT_TRUE(recording->last_options.has_value());
  EXPECT_EQ(recording->last_options->optimization_level, 2);
  EXPECT_TRUE(recording->last_options->deterministic);
  EXPECT_FALSE(recording->last_options->fast_math);
}

TEST(MusaGpuCompilerTest, RejectsRelocatableCompilationBeforeProviderCall) {
  auto provider = std::make_unique<RecordingCompilationProvider>();
  RecordingCompilationProvider* recording = provider.get();
  TestMusaGpuCompiler compiler(std::move(provider));
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module = ElementalModule(context);
  ASSERT_NE(module, nullptr);

  EXPECT_THAT(
      compiler.CompileTargetBinary(
          HloModuleConfig(), module.get(), MusaDevice(),
          /*relocatable=*/true, /*debug_module=*/nullptr,
          /*shard_number=*/std::nullopt),
      StatusIs(absl::StatusCode::kUnimplemented, HasSubstr("relocatable")));
  EXPECT_EQ(recording->compile_count, 0);
}

TEST(MusaGpuCompilerTest, SkipsEmptySharedGpuConstantsModule) {
  auto provider = std::make_unique<RecordingCompilationProvider>();
  RecordingCompilationProvider* recording = provider.get();
  TestMusaGpuCompiler compiler(std::move(provider));
  llvm::LLVMContext context;
  llvm::Module module("constants", context);
  module.setTargetTriple(llvm::Triple(musa::TargetTriple()));
  module.setDataLayout(musa::DataLayout());

  auto result = compiler.CompileTargetBinary(
      HloModuleConfig(), &module, MusaDevice(), /*relocatable=*/false,
      /*debug_module=*/nullptr, /*shard_number=*/std::nullopt);
  ASSERT_THAT(result, IsOk());
  EXPECT_TRUE(result->binary.empty());
  EXPECT_EQ(recording->compile_count, 0);
}

TEST(MusaGpuCompilerTest, WiresDedicatedMublasGemmRewrite) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(R"(
HloModule musa_gemm_rewrite

ENTRY main {
  lhs = f32[8,16]{1,0} parameter(0)
  rhs = f32[16,4]{1,0} parameter(1)
  ROOT result = f32[8,4]{1,0} dot(lhs, rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)"));

  TestMusaGpuCompiler compiler(
      std::make_unique<RecordingCompilationProvider>());
  HloPassPipeline pipeline("musa-gemm-rewriter");
  compiler.AddGemmRewriterPasses(pipeline, module->config().debug_options(),
                                 MusaDevice().gpu_compute_capability(),
                                 se::SemanticVersion{4, 0, 1});
  ASSERT_OK_AND_ASSIGN(bool changed, pipeline.Run(module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_EQ(root->opcode(), HloOpcode::kCustomCall);
  EXPECT_EQ(root->custom_call_target(), "__mublas$gemm");
  EXPECT_TRUE(root->shape().IsArray());
}

TEST(MusaGpuCompilerTest,
     RewritesTriangularSolveAndRequiresExactMublasTrsmAbi) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(R"(
HloModule musa_triangular_solve_rewrite

ENTRY main {
  a = f32[4,4]{0,1} parameter(0)
  b = f32[3,4]{0,1} parameter(1)
  ROOT solve = f32[3,4]{0,1} triangular-solve(a, b), lower=true,
    transpose_a=TRANSPOSE
}
)"));

  stream_executor::GpuTargetConfigProto target_proto;
  *target_proto.mutable_gpu_device_info() = MusaDevice().ToProto();
  target_proto.set_platform_name("MUSA");
  target_proto.set_device_description_str("MTT S80");
  ASSERT_OK_AND_ASSIGN(GpuTargetConfig target_config,
                       GpuTargetConfig::FromProto(target_proto));

  TestMusaGpuCompiler compiler(
      std::make_unique<RecordingCompilationProvider>());
  Compiler::CompileOptions options;
  GpuAliasInfo alias_info(MusaDevice());
  mlir::MLIRContext mlir_context;
  std::unique_ptr<CompilationStats> compilation_stats =
      CompilationStats::MakeNoopStats();
  ASSERT_OK(compiler.OptimizeHloPostLayoutAssignment(
      module.get(), /*stream_exec=*/nullptr, options, target_config,
      &alias_info, /*thread_pool=*/nullptr, compilation_stats.get(),
      &mlir_context));

  const HloInstruction* triangular_solve = nullptr;
  for (const HloInstruction* instruction :
       module->entry_computation()->instructions()) {
    if (instruction->opcode() == HloOpcode::kCustomCall &&
        instruction->custom_call_target() == "__cublas$triangularSolve") {
      triangular_solve = instruction;
      break;
    }
  }
  ASSERT_NE(triangular_solve, nullptr);
  EXPECT_TRUE(triangular_solve->shape().IsTuple());

  ASSERT_OK_AND_ASSIGN(
      stream_executor::ExecutableAbiVersion version,
      compiler.CreateExecutableAbiVersion(*module, MusaDevice(), {}));
  const auto& libraries =
      version.proto().musa_platform_version().required_optional_library_abis();
  ASSERT_EQ(libraries.size(), 2);
  EXPECT_EQ(libraries[0].name(),
            stream_executor::musa::kMusaMuBlasLibraryAbiName);
  EXPECT_EQ(libraries[0].abi_version(),
            stream_executor::musa::kMusaMuBlasLibraryAbiVersion);
  EXPECT_TRUE(libraries[0].fingerprint().empty());
  EXPECT_EQ(libraries[1].name(),
            stream_executor::musa::kMusaMuBlasTrsmLibraryAbiName);
  EXPECT_EQ(libraries[1].abi_version(),
            stream_executor::musa::kMusaMuBlasTrsmLibraryAbiVersion);
  EXPECT_EQ(libraries[1].fingerprint(),
            stream_executor::musa::kMusaMuBlasTrsmAbiFingerprintV1);
}

TEST(MusaGpuCompilerTest, LeavesStandardScanAndSortOnSharedHloFallbacks) {
  constexpr absl::string_view hlo = R"(
HloModule no_musa_cub

add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  sum = f32[] add(lhs, rhs)
  ROOT result = (f32[], f32[]) tuple(sum, sum)
}

less_than {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT result = pred[] compare(lhs, rhs), direction=LT
}

ENTRY main {
  scan_input = f32[100] parameter(0)
  zero = f32[] constant(0)
  scan = (f32[100], f32[]) scan(scan_input, zero), dimensions={0}, num_carries=1, is_associative=true, to_apply=add
  scanned = f32[100] get-tuple-element(scan), index=0
  sort_input = f32[32768] parameter(1)
  sorted = f32[32768] sort(sort_input), dimensions={0}, to_apply=less_than
  ROOT result = (f32[100], f32[32768]) tuple(scanned, sorted)
}
)";

  HloModuleConfig config;
  config.mutable_debug_options().set_xla_gpu_enable_cub_radix_sort(true);
  auto module = ParseAndReturnUnverifiedModule(hlo, config);
  ASSERT_THAT(module, IsOk());

  stream_executor::GpuTargetConfigProto target_proto;
  *target_proto.mutable_gpu_device_info() = MusaDevice().ToProto();
  target_proto.set_platform_name("MUSA");
  target_proto.set_device_description_str("MTT S80");
  auto target_config = gpu::GpuTargetConfig::FromProto(target_proto);
  ASSERT_TRUE(target_config.ok()) << target_config.status();

  Compiler::CompileOptions options;
  options.gpu_topology = GetSingleDeviceGpuTopology(
      /*platform_version=*/"", *target_config);
  options.early_exit_with_layouts = true;

  TestMusaGpuCompiler compiler(
      std::make_unique<RecordingCompilationProvider>());
  auto optimized = compiler.RunHloPasses(std::move(*module),
                                         /*executor=*/nullptr, options);
  ASSERT_THAT(optimized, IsOk());

  bool found_reduce_window = false;
  bool found_sort = false;
  bool found_call = false;
  std::vector<std::string> cub_custom_calls;
  for (const HloComputation* computation : (*optimized)->computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      found_reduce_window |= instruction->opcode() == HloOpcode::kReduceWindow;
      found_sort |= instruction->opcode() == HloOpcode::kSort;
      found_call |= instruction->opcode() == HloOpcode::kCall;
      if (instruction->opcode() == HloOpcode::kCustomCall &&
          absl::StartsWith(instruction->custom_call_target(),
                           "xla.gpu.ext.cub_")) {
        cub_custom_calls.push_back(instruction->custom_call_target());
      }
    }
  }
  EXPECT_TRUE(found_reduce_window);
  EXPECT_TRUE(found_sort);
  EXPECT_FALSE(found_call);
  EXPECT_THAT(cub_custom_calls, IsEmpty());
}

}  // namespace
}  // namespace xla::gpu
