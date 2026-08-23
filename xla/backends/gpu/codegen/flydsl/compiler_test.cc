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
#include "xla/backends/gpu/codegen/flydsl/compiler.h"

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "gtest/gtest.h"

namespace xla::gpu::flydsl {
namespace {

TEST(FlyDslCompilerTest, RegistersNativeDialects) {
  mlir::DialectRegistry registry;
  RegisterDialects(registry);
  mlir::MLIRContext context(registry);

  EXPECT_NE(context.getOrLoadDialect<mlir::fly::FlyDialect>(), nullptr);
  EXPECT_NE(context.getOrLoadDialect<mlir::fly_rocdl::FlyROCDLDialect>(),
            nullptr);
}

TEST(FlyDslCompilerTest, EmptyModuleHasNoFlyOperations) {
  mlir::DialectRegistry registry;
  RegisterDialects(registry);
  mlir::MLIRContext context(registry);
  mlir::OpBuilder builder(&context);
  mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());

  EXPECT_FALSE(HasOperations(module));
}

TEST(FlyDslCompilerTest, DetectsFlyOperations) {
  mlir::DialectRegistry registry;
  RegisterDialects(registry);
  mlir::MLIRContext context(registry);
  mlir::OpBuilder builder(&context);
  mlir::ModuleOp module = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(module.getBody());
  mlir::OperationState state(builder.getUnknownLoc(), "fly.static");
  builder.create(state);

  EXPECT_TRUE(HasOperations(module));
}

TEST(FlyDslCompilerTest, LowersBf16MfmaToRocdl) {
  mlir::DialectRegistry registry;
  RegisterDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::gpu::GPUDialect, mlir::LLVM::LLVMDialect,
                  mlir::ROCDL::ROCDLDialect, mlir::scf::SCFDialect,
                  mlir::vector::VectorDialect>();
  mlir::MLIRContext context(registry);
  constexpr llvm::StringLiteral kModule = R"mlir(
    module {
      func.func @bf16_mfma(
          %a: vector<4xbf16>, %b: vector<4xbf16>,
          %c: vector<16xf32>) -> vector<16xf32> {
        %atom = fly.make_mma_atom :
          !fly.mma_atom<!fly_rocdl.cdna3.mfma<32x32x8, (bf16, bf16) -> f32>>
        %result = fly.mma_atom_call_ssa(%atom, %a, %b, %c) :
          (!fly.mma_atom<!fly_rocdl.cdna3.mfma<32x32x8, (bf16, bf16) -> f32>>,
           vector<4xbf16>, vector<4xbf16>, vector<16xf32>) -> vector<16xf32>
        return %result : vector<16xf32>
      }
    }
  )mlir";
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kModule, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(HasOperations(*module));

  mlir::PassManager pm(&context);
  AddLoweringPasses(pm);
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  bool found_mfma = false;
  module->walk([&](mlir::Operation* op) {
    if (op->getName().getStringRef().starts_with("rocdl.mfma")) {
      found_mfma = true;
    }
  });
  EXPECT_TRUE(found_mfma);
  EXPECT_FALSE(HasOperations(*module));
}

TEST(FlyDslCompilerTest, WrapsKernelArgumentMemoryInFlyAndLowersBackToLlvm) {
  mlir::DialectRegistry registry;
  RegisterDialects(registry);
  registry.insert<mlir::LLVM::LLVMDialect, mlir::ROCDL::ROCDLDialect>();
  mlir::MLIRContext context(registry);
  constexpr llvm::StringLiteral kModule = R"mlir(
    module {
      llvm.func @kernel(%input: !llvm.ptr, %output: !llvm.ptr) {
        %value = llvm.load %input invariant {alignment = 4 : i64} : !llvm.ptr -> f32
        llvm.store %value, %output : f32, !llvm.ptr
        llvm.return
      }
    }
  )mlir";
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kModule, &context);
  ASSERT_TRUE(module);

  MarkGenericFusion(*module);
  EXPECT_TRUE(IsGenericFusion(*module));

  mlir::PassManager wrap_pm(&context);
  AddGenericMemoryPasses(wrap_pm);
  ASSERT_TRUE(mlir::succeeded(wrap_pm.run(*module)));
  EXPECT_TRUE(HasOperations(*module));

  int fly_loads = 0;
  int fly_stores = 0;
  module->walk([&](mlir::fly::PtrLoadOp) { ++fly_loads; });
  module->walk([&](mlir::fly::PtrStoreOp) { ++fly_stores; });
  EXPECT_EQ(fly_loads, 1);
  EXPECT_EQ(fly_stores, 1);

  mlir::PassManager lower_pm(&context);
  AddLoweringPasses(lower_pm, /*restore_generic_memory_metadata=*/true);
  ASSERT_TRUE(mlir::succeeded(lower_pm.run(*module)));
  EXPECT_FALSE(HasOperations(*module));

  int llvm_loads = 0;
  int llvm_stores = 0;
  module->walk([&](mlir::LLVM::LoadOp load) {
    ++llvm_loads;
    EXPECT_TRUE(load.getInvariant());
    EXPECT_EQ(load.getAlignment(), 4);
  });
  module->walk([&](mlir::LLVM::StoreOp) { ++llvm_stores; });
  EXPECT_EQ(llvm_loads, 1);
  EXPECT_EQ(llvm_stores, 1);
}

}  // namespace
}  // namespace xla::gpu::flydsl
