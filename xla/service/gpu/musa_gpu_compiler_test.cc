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
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "xla/service/gpu/musa/musa_compilation_provider.h"
#include "xla/service/gpu/musa/musa_llvm14_compatibility.h"
#include "xla/service/gpu/musa/musa_shim_abi.h"
#include "xla/service/gpu/target_constants.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/musa/musa_compute_capability.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

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
  musa::MusaCompilationCapabilities capabilities() const override {
    return {};
  }
  absl::string_view name() const override { return "recording"; }

  mutable int compile_count = 0;
  mutable std::optional<musa::MusaLlvm14CompatibilityResult> last_module;
  mutable std::optional<musa::MusaCompilationOptions> last_options;

 private:
  musa::MusaCompilationIdentity identity_;
};

class TestMusaGpuCompiler : public MusaGpuCompiler {
 public:
  using MusaGpuCompiler::MusaGpuCompiler;
  using MusaGpuCompiler::CompileTargetBinary;
};

std::unique_ptr<llvm::Module> ElementalModule(llvm::LLVMContext& context) {
  const std::string source = absl::StrCat(
      "source_filename = \"c10-test\"\n",
      "target datalayout = \"", musa::DataLayout(), "\"\n",
      "target triple = \"", musa::TargetTriple(), "\"\n\n",
      "define void @kernel(ptr addrspace(1) %out) #0 {\n",
      "entry:\n",
      "  store i32 7, ptr addrspace(1) %out, align 4\n",
      "  ret void\n",
      "}\n\n",
      "attributes #0 = { \"", musa::kMusaLlvmKernelMarker, "\" }\n");
  llvm::SMDiagnostic diagnostic;
  return llvm::parseAssemblyString(source, diagnostic, context);
}

stream_executor::DeviceDescription MusaDevice() {
  stream_executor::DeviceDescription device;
  device.set_gpu_compute_capability(stream_executor::GpuComputeCapability(
      stream_executor::MusaComputeCapability("mp_21", 2, 1,
                                             /*hardware_warp_size=*/128,
                                             /*logical_subgroup_size=*/32)));
  return device;
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

  EXPECT_THAT(compiler.CompileTargetBinary(
                  HloModuleConfig(), module.get(), MusaDevice(),
                  /*relocatable=*/true, /*debug_module=*/nullptr,
                  /*shard_number=*/std::nullopt),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("relocatable")));
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

}  // namespace
}  // namespace xla::gpu
