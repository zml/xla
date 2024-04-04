/* Copyright 2024 The OpenXLA Authors.

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
#include "xla/service/gpu/fusions/mlir/mlir_fusion_emitter.h"

#include <cstdint>
#include <optional>
#include <string>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"  // from @llvm-project
#include "mlir/Dialect/Complex/IR/Complex.h"  // from @llvm-project
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/Math/IR/Math.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/service/gpu/fusions/mlir/computation_partitioner.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/filecheck.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

class DummyCopyFusionEmitter : public MlirFusionEmitterBase {
 public:
  LaunchDimensions launch_dimensions() const final { return {1, 100}; }

  std::optional<IndexingMap> ComputeThreadIdToOutputIndexing(
      int64_t, mlir::MLIRContext*) const final {
    return std::nullopt;
  }

  std::optional<IndexingMap> ComputeThreadIdToInputIndexing(
      int64_t, int64_t, mlir::MLIRContext*) const final {
    return std::nullopt;
  }

 protected:
  absl::Status EmitEntryFunction(
      const mlir_converter::PartitionedComputations& computations,
      const mlir_converter::CallTargetProvider& call_targets,
      mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion) const {
    mlir::ImplicitLocOpBuilder b(entry_function.getLoc(), entry_function);
    b.setInsertionPointToStart(entry_function.addEntryBlock());
    auto thread_id = EmitThreadId(b, 0);
    auto value = b.create<mlir::tensor::ExtractOp>(
        entry_function.getArgument(0), mlir::ValueRange{thread_id});
    auto result = b.create<mlir::tensor::InsertOp>(
        value, entry_function.getArgument(1), mlir::ValueRange{thread_id});
    b.create<mlir::func::ReturnOp>(result->getResults());
    return absl::OkStatus();
  }
};

class MlirFusionEmitterTest : public HloTestBase {
 protected:
  MlirFusionEmitterTest() {
    context_.loadDialect<mlir::tensor::TensorDialect, mlir::func::FuncDialect,
                         mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                         mlir::complex::ComplexDialect, mlir::math::MathDialect,
                         mlir::scf::SCFDialect, mlir::mhlo::MhloDialect,
                         mlir::gpu::GPUDialect, mlir::NVVM::NVVMDialect>();
    mlir::DialectRegistry registry;
    mlir::func::registerInlinerExtension(registry);
    mlir::registerBuiltinDialectTranslation(registry);
    mlir::registerLLVMDialectTranslation(registry);
    mlir::registerNVVMDialectTranslation(registry);
    context_.appendDialectRegistry(registry);
  }

  mlir::MLIRContext context_;
  stream_executor::DeviceDescription device_info_ =
      TestGpuDeviceInfo::RTXA6000DeviceInfo();
};

constexpr absl::string_view kModule = R"(
    fused_computation {
      ROOT %p0 = f32[100] parameter(0)
    }

    ENTRY main {
      %p0 = f32[100] parameter(0)
      ROOT fusion = f32[100] fusion(%p0), kind=kLoop, calls=fused_computation
    })";

TEST_F(MlirFusionEmitterTest, CreateMlirModule) {
  auto module = ParseAndReturnVerifiedModule(kModule).value();
  DummyCopyFusionEmitter emitter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto mlir_module,
      emitter.CreateMLIRModule(
          context_,
          *Cast<HloFusionInstruction>(
              module->entry_computation()->root_instruction()),
          "fusion",
          /*buffer_assignment=*/nullptr));

  std::string out;
  llvm::raw_string_ostream stream(out);
  stream << *mlir_module;

  TF_ASSERT_OK_AND_ASSIGN(auto filecheck_result, RunFileCheck(out, R"(
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

TEST_F(MlirFusionEmitterTest, CreateLLVMModule) {
  llvm::LLVMContext llvm_context;

  auto module = ParseAndReturnVerifiedModule(kModule).value();
  DummyCopyFusionEmitter emitter;
  TF_ASSERT_OK_AND_ASSIGN(
      auto llvm_module,
      emitter.CreateLLVMModule(
          context_, llvm_context, device_info_,
          *Cast<HloFusionInstruction>(
              module->entry_computation()->root_instruction()),
          "fusion",
          /*buffer_assignment=*/nullptr));

  std::string out;
  llvm::raw_string_ostream stream(out);
  stream << *llvm_module;

  TF_ASSERT_OK_AND_ASSIGN(auto filecheck_result, RunFileCheck(out, R"(
    // CHECK: define void @fusion(ptr noalias %[[IN:.*]], ptr noalias %[[OUT:.*]])
    // CHECK:   %[[TID:.*]] = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
    // CHECK:   %[[EXT:.*]] = sext i32 %[[TID]] to i64
    // CHECK:   %[[TRUNC:.*]] = trunc i64 %[[EXT]] to i32
    // CHECK:   %[[IN_PTR:.*]] = getelementptr inbounds float, ptr %[[IN]], i32 %[[TRUNC]]
    // CHECK:   %[[VAL:.*]] = load float, ptr %[[IN_PTR]], align 4
    // CHECK:   %[[OUT_PTR:.*]] = getelementptr inbounds float, ptr %[[OUT]], i32 %[[TRUNC]]
    // CHECK:   store float %[[VAL]], ptr %[[OUT_PTR]], align 4
    // CHECK:   ret void
  )"));
  EXPECT_TRUE(filecheck_result);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
