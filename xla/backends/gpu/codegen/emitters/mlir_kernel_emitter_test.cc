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
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "xla/backends/gpu/codegen/cubin_custom_kernel_compiler.h"
#include "xla/backends/gpu/codegen/fusions.h"
#include "xla/backends/gpu/codegen/kernel_compiler.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/emitters/transforms/musa_gpu_to_llvm.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/runtime/object_pool.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/musa/musa_llvm14_compatibility.h"
#include "xla/service/gpu/musa/musa_shim_abi.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/musa/musa_compute_capability.h"
#include "xla/stream_executor/musa/musa_target_contract.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

class DummyCopyEmitter : public MlirKernelEmitter {
 public:
  LaunchDimensions launch_dimensions() const final { return {1, 100}; }

  std::optional<IndexingMap> ComputeThreadIdToOutputIndexing(
      int64_t, mlir::MLIRContext*) const final {
    return std::nullopt;
  }

  std::optional<std::vector<IndexingMap>> ComputeThreadIdToInputIndexing(
      int64_t, mlir::MLIRContext*) const final {
    return std::nullopt;
  }

 protected:
  absl::Status EmitEntryFunction(
      const emitters::PartitionedComputations& computations,
      const emitters::CallTargetProvider& call_targets,
      mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion) const override {
    mlir::ImplicitLocOpBuilder b(entry_function.getLoc(), entry_function);
    b.setInsertionPointToStart(entry_function.addEntryBlock());
    auto thread_id = EmitThreadId(b, 0);
    auto value = mlir::tensor::ExtractOp::create(
        b, entry_function.getArgument(0), mlir::ValueRange{thread_id});
    auto result = mlir::tensor::InsertOp::create(
        b, value, entry_function.getArgument(1), mlir::ValueRange{thread_id});
    mlir::func::ReturnOp::create(b, result->getResults());
    return absl::OkStatus();
  }
};

class MlirKernelFusionTest : public HloHardwareIndependentTestBase {
 protected:
  MlirKernelFusionTest() {
    mlir_context_.appendDialectRegistry(
        MlirKernelEmitter::GetDialectRegistry());
    mlir_context_.loadAllAvailableDialects();
  }

  mlir::MLIRContext mlir_context_;
  stream_executor::DeviceDescription device_info_ =
      TestGpuDeviceInfo::CudaOrRocmDeviceInfo();
};

constexpr absl::string_view kModule = R"(
    fused_computation {
      ROOT %p0 = f32[100] parameter(0)
    }

    ENTRY main {
      %p0 = f32[100] parameter(0)
      ROOT fusion = f32[100] fusion(%p0), kind=kLoop, calls=fused_computation
    })";

stream_executor::DeviceDescription MusaS80DeviceInfo() {
  stream_executor::DeviceDescription device =
      TestGpuDeviceInfo::RTXA6000DeviceInfo();
  device.set_gpu_compute_capability(stream_executor::GpuComputeCapability(
      stream_executor::MusaComputeCapability(
          stream_executor::musa::kS80TargetArchitecture,
          stream_executor::musa::kS80ComputeCapabilityMajor,
          stream_executor::musa::kS80ComputeCapabilityMinor,
          stream_executor::musa::kS80HardwareWarpSize,
          stream_executor::musa::kS80CompilerLogicalSubgroupSize)));
  device.set_threads_per_warp(stream_executor::musa::kS80HardwareWarpSize);
  return device;
}

TEST_F(MlirKernelFusionTest, CreateMlirModule) {
  auto module = ParseAndReturnVerifiedModule(kModule).value();
  DummyCopyEmitter emitter;
  ASSERT_OK_AND_ASSIGN(auto mlir_module,
                       emitter.CreateMLIRModule(
                           mlir_context_,
                           *Cast<HloFusionInstruction>(
                               module->entry_computation()->root_instruction()),
                           "fusion",
                           /*buffer_assignment=*/nullptr));

  std::string out;
  llvm::raw_string_ostream stream(out);
  stream << *mlir_module;

  ASSERT_OK_AND_ASSIGN(auto filecheck_result, RunFileCheck(out, R"(
    // CHECK:      func.func @fusion(
    // CHECK-SAME:     %[[IN:.*]]: tensor<100xf32> {xla.slice_index = 0
    // CHECK-SAME:     %[[OUT:.*]]: tensor<100xf32> {xla.slice_index = 1
    // CHECK:        %[[TID:.*]] = gpu.thread_id x
    // CHECK:        %[[VAL:.*]] = tensor.extract %[[IN]][%[[TID]]]
    // CHECK:        %[[RET:.*]] = tensor.insert %[[VAL]]
    // CHECK-SAME:     into %[[OUT]][%[[TID]]]
    // CHECK:        return %[[RET]]
  )"));
  EXPECT_TRUE(filecheck_result);
}

TEST_F(MlirKernelFusionTest, CreateLLVMModule) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kModule));
  CubinCustomKernelCompiler kernel_compiler(
      [](llvm::Module& llvm_module, const se::DeviceDescription& descr,
         const DebugOptions& opts) { return std::vector<uint8_t>{}; },
      device_info_, module->config().debug_options());

  ObjectPool<std::unique_ptr<mlir::MLIRContext>> mlir_context_pool(
      []() { return CreateMlirContext(); });
  MlirKernelFusion emitter(std::make_unique<DummyCopyEmitter>());
  ASSERT_OK_AND_ASSIGN(BorrowedMlirContext borrowed_context,
                       mlir_context_pool.GetOrCreate());
  ASSERT_OK_AND_ASSIGN(
      LlvmKernelSource source,
      emitter
          .CreateLLVMModule(
              device_info_,
              *Cast<HloFusionInstruction>(
                  module->entry_computation()->root_instruction()),
              "fusion",
              /*buffer_assignment=*/nullptr, &kernel_compiler,
              std::move(borrowed_context))
          .Await());
  auto llvm_module = std::move(source).thread_safe_module();

  std::string out;
  llvm::raw_string_ostream stream(out);
  stream << *llvm_module.getModuleUnlocked();

  ASSERT_OK_AND_ASSIGN(
      auto filecheck_result,
      RunFileCheck(out,
                   absl::StrReplaceAll(
                       R"(
    // CHECK: define void @fusion(ptr noalias %[[IN:.*]], ptr noalias %[[OUT:.*]])
    // CHECK:   %[[TID:.*]] = call i32 TIDX()
    // CHECK:   %[[IN_PTR:.*]] = getelementptr inbounds [100 x float], ptr %[[IN]], i32 0, i32 %[[TID]]
    // CHECK:   %[[VAL:.*]] = load float, ptr %[[IN_PTR]], align 4
    // CHECK:   %[[OUT_PTR:.*]] = getelementptr inbounds [100 x float], ptr %[[OUT]], i32 0, i32 %[[TID]]
    // CHECK:   store float %[[VAL]], ptr %[[OUT_PTR]], align 4
    // CHECK:   ret void
  )",
                       {{"TIDX", device_info_.gpu_compute_capability().IsRocm()
                                     ? "@llvm.amdgcn.workitem.id.x"
                                     : "@llvm.nvvm.read.ptx.sreg.tid.x"}})));
  EXPECT_TRUE(filecheck_result);
}

TEST_F(MlirKernelFusionTest, SelectsMusaMlirKernelFusion) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kModule));
  stream_executor::DeviceDescription device = MusaS80DeviceInfo();
  const HloInstruction* root = module->entry_computation()->root_instruction();
  HloFusionAnalysis analysis = HloFusionAnalysis::Create(*root, device);

  std::unique_ptr<FusionInterface> emitter =
      GetFusionEmitter(PreBufferAssignmentFusionInfo{analysis});
  EXPECT_NE(dynamic_cast<MusaMlirKernelFusion*>(emitter.get()), nullptr);
}

TEST_F(MlirKernelFusionTest, MusaFusionQualificationFailsClosed) {
  using Kind = HloFusionAnalysis::EmitterFusionKind;
  EXPECT_TRUE(MusaFusionEmitterQualification(Kind::kLoop).IsAllowed());
  EXPECT_TRUE(MusaFusionEmitterQualification(Kind::kReduction).IsAllowed());
  EXPECT_TRUE(MusaFusionEmitterQualification(Kind::kScatter).IsAllowed());
  EXPECT_TRUE(MusaFusionEmitterQualification(Kind::kTranspose).IsAllowed());
  EXPECT_TRUE(MusaFusionEmitterQualification(Kind::kConcatenate).IsAllowed());
  EXPECT_TRUE(
      MusaFusionEmitterQualification(Kind::kCustomFusion).IsForbidden());
  EXPECT_TRUE(MusaFusionEmitterQualification(Kind::kTriton).IsForbidden());
  EXPECT_TRUE(MusaFusionEmitterQualification(Kind::kCuDnn).IsForbidden());
  EXPECT_TRUE(MusaFusionEmitterQualification(Kind::kSort).IsForbidden());
}

TEST_F(MlirKernelFusionTest, MusaLLVMModuleUsesShimAbiAndKernelMarker) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kModule));
  stream_executor::DeviceDescription device = MusaS80DeviceInfo();
  CubinCustomKernelCompiler kernel_compiler(
      [](llvm::Module&, const se::DeviceDescription&, const DebugOptions&) {
        return std::vector<uint8_t>{};
      },
      device, module->config().debug_options());

  ObjectPool<std::unique_ptr<mlir::MLIRContext>> mlir_context_pool(
      []() { return CreateMlirContext(); });
  MusaMlirKernelFusion emitter(std::make_unique<DummyCopyEmitter>());
  ASSERT_OK_AND_ASSIGN(BorrowedMlirContext borrowed_context,
                       mlir_context_pool.GetOrCreate());
  ASSERT_OK_AND_ASSIGN(
      LlvmKernelSource source,
      emitter
          .CreateLLVMModule(
              device,
              *Cast<HloFusionInstruction>(
                  module->entry_computation()->root_instruction()),
              "fusion", /*buffer_assignment=*/nullptr, &kernel_compiler,
              std::move(borrowed_context))
          .Await());
  auto llvm_module = std::move(source).thread_safe_module();
  llvm::Module* raw_module = llvm_module.getModuleUnlocked();

  std::string verifier_error;
  llvm::raw_string_ostream verifier_stream(verifier_error);
  EXPECT_FALSE(llvm::verifyModule(*raw_module, &verifier_stream))
      << verifier_error;

  EXPECT_EQ(raw_module->getTargetTriple().str(),
            stream_executor::musa::kMusaTargetTriple);
  EXPECT_EQ(raw_module->getDataLayoutStr(),
            stream_executor::musa::kMusaTargetDataLayout);

  llvm::Function* kernel = raw_module->getFunction("fusion");
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(kernel->getCallingConv(), llvm::CallingConv::C);
  EXPECT_TRUE(kernel->hasFnAttribute(::xla::emitters::kMusaKernelMarker));
  EXPECT_NE(kernel->getCallingConv(),
            stream_executor::musa::kMusaKernelCallingConvention);
  EXPECT_FALSE(kernel->hasFnAttribute("nvvm.reqntid"));
  EXPECT_FALSE(kernel->hasFnAttribute("nvvm.maxntid"));
  EXPECT_FALSE(kernel->hasFnAttribute("amdgpu-flat-work-group-size"));
  EXPECT_FALSE(kernel->hasFnAttribute("amdgpu-max-num-workgroups"));
  EXPECT_EQ(raw_module->getNamedMetadata("nvvm.annotations"), nullptr);
  EXPECT_EQ(raw_module->getNamedMetadata("amdgpu.annotations"), nullptr);

  for (llvm::Argument& argument : kernel->args()) {
    auto* pointer_type = llvm::dyn_cast<llvm::PointerType>(argument.getType());
    if (pointer_type == nullptr) continue;
    const auto* address_space =
        ::xla::gpu::musa::FindMusaAddressSpace(pointer_type->getAddressSpace());
    ASSERT_NE(address_space, nullptr);
    EXPECT_TRUE(address_space->allowed_in_interchange);
  }

  llvm::Function* shim = raw_module->getFunction("__xla_musa_v1_read_tid_x");
  ASSERT_NE(shim, nullptr);
  EXPECT_TRUE(shim->isDeclaration());
  EXPECT_TRUE(shim->hasFnAttribute(llvm::Attribute::NoUnwind));
  EXPECT_FALSE(shim->hasFnAttribute(llvm::Attribute::Convergent));
  EXPECT_EQ(shim->getMemoryEffects(), llvm::MemoryEffects::none());

  int shim_calls = 0;
  for (llvm::BasicBlock& block : *kernel) {
    for (llvm::Instruction& instruction : block) {
      auto* call = llvm::dyn_cast<llvm::CallBase>(&instruction);
      if (call == nullptr) continue;
      EXPECT_FALSE(call->isInlineAsm());
      if (call->getCalledFunction() == shim) {
        ++shim_calls;
        EXPECT_TRUE(call->getAttributes().isEmpty());
      }
    }
  }
  EXPECT_EQ(shim_calls, 1);

  for (llvm::Function& function : *raw_module) {
    EXPECT_FALSE(function.getName().starts_with("llvm.nvvm."));
    EXPECT_FALSE(function.getName().starts_with("llvm.amdgcn."));
    EXPECT_FALSE(function.getName().starts_with("llvm.musa."));
    EXPECT_NE(function.getCallingConv(),
              stream_executor::musa::kMusaKernelCallingConvention);
    if (function.getName().starts_with(
            ::xla::gpu::musa::kMusaShimSymbolPrefix)) {
      EXPECT_NE(::xla::gpu::musa::FindMusaShim(function.getName().str()),
                nullptr);
    }
  }
  for (llvm::NamedMDNode& metadata : raw_module->named_metadata()) {
    EXPECT_FALSE(metadata.getName().starts_with("nvvm."));
    EXPECT_FALSE(metadata.getName().starts_with("amdgpu."));
    EXPECT_FALSE(metadata.getName().starts_with("rocdl."));
  }

  ASSERT_OK_AND_ASSIGN(
      ::xla::gpu::musa::MusaLlvm14CompatibilityResult compatibility,
      ::xla::gpu::musa::NormalizeMusaLlvmForLlvm14(*raw_module, "fusion"));
  EXPECT_EQ(compatibility.metadata.kernel_entry_names,
            std::vector<std::string>{"fusion"});
  EXPECT_TRUE(compatibility.metadata.exported_globals.empty());
  EXPECT_FALSE(absl::StrContains(compatibility.normalized_llvm,
                                 ::xla::gpu::musa::kMusaLlvmKernelMarker));
  EXPECT_FALSE(absl::StrContains(compatibility.normalized_llvm, "memory("));
  EXPECT_THAT(::xla::gpu::musa::ValidateMusaBridgeIr(
                  compatibility.normalized_llvm, compatibility.metadata),
              tsl::testing::IsOk());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
