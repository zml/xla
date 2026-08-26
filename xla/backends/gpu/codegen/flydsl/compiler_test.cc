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
#include "mlir/Dialect/Math/IR/Math.h"
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

TEST(FlyDslCompilerTest, LowersNativeF32ErfToOcmlBeforeExpansion) {
  mlir::DialectRegistry registry;
  RegisterDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::gpu::GPUDialect,
                  mlir::LLVM::LLVMDialect, mlir::math::MathDialect,
                  mlir::ROCDL::ROCDLDialect, mlir::vector::VectorDialect>();
  mlir::MLIRContext context(registry);
  constexpr llvm::StringLiteral kModule = R"mlir(
    module {
      gpu.module @kernels {
        gpu.func @kernel(%value: f32) kernel {
          %result = math.erf %value : f32
          gpu.return
        }
      }
    }
  )mlir";
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kModule, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  AddLoweringPasses(pm);
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  int erf_calls = 0;
  int math_erfs = 0;
  module->walk([&](mlir::LLVM::CallOp call) {
    if (call.getCallee() == "__ocml_erf_f32") {
      ++erf_calls;
    }
  });
  module->walk([&](mlir::math::ErfOp) { ++math_erfs; });
  EXPECT_EQ(erf_calls, 1);
  EXPECT_EQ(math_erfs, 0);
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
        llvm.store %value, %output {alignment = 4 : i64} : f32, !llvm.ptr
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
  module->walk([&](mlir::LLVM::StoreOp store) {
    ++llvm_stores;
    EXPECT_EQ(store.getAlignment(), 4);
  });
  EXPECT_EQ(llvm_loads, 1);
  EXPECT_EQ(llvm_stores, 1);
}

TEST(FlyDslCompilerTest, UsesRawBuffersForBoundedKernelArguments) {
  mlir::DialectRegistry registry;
  RegisterDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::LLVM::LLVMDialect,
                  mlir::ROCDL::ROCDLDialect>();
  mlir::MLIRContext context(registry);
  constexpr llvm::StringLiteral kModule = R"mlir(
    module {
      llvm.func @kernel(
          %input: !llvm.ptr {llvm.dereferenceable = 4096 : index},
          %output: !llvm.ptr {llvm.dereferenceable = 4096 : index}) {
        %index = llvm.mlir.constant(3 : i64) : i64
        %input_element = llvm.getelementptr %input[%index] :
          (!llvm.ptr, i64) -> !llvm.ptr, f32
        %output_element = llvm.getelementptr %output[%index] :
          (!llvm.ptr, i64) -> !llvm.ptr, f32
        %value = llvm.load %input_element : !llvm.ptr -> f32
        llvm.store %value, %output_element : f32, !llvm.ptr
        llvm.return
      }
    }
  )mlir";
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kModule, &context);
  ASSERT_TRUE(module);

  mlir::PassManager wrap_pm(&context);
  AddGenericMemoryPasses(wrap_pm);
  ASSERT_TRUE(mlir::succeeded(wrap_pm.run(*module)));

  int make_pointers = 0;
  int add_offsets = 0;
  module->walk([&](mlir::fly::MakePtrOp op) {
    ++make_pointers;
    auto pointer_type = op.getResult().getType();
    EXPECT_TRUE(mlir::fly::isTargetAddressSpace<
                mlir::fly_rocdl::BufferDescAddressAttr>(
        pointer_type.getAddressSpace()));
  });
  module->walk([&](mlir::fly::AddOffsetOp) { ++add_offsets; });
  EXPECT_EQ(make_pointers, 2);
  EXPECT_EQ(add_offsets, 2);

  mlir::PassManager lower_pm(&context);
  AddLoweringPasses(lower_pm, /*restore_generic_memory_metadata=*/true);
  ASSERT_TRUE(mlir::succeeded(lower_pm.run(*module)));
  EXPECT_FALSE(HasOperations(*module));

  int raw_loads = 0;
  int raw_stores = 0;
  module->walk([&](mlir::ROCDL::RawPtrBufferLoadOp) { ++raw_loads; });
  module->walk([&](mlir::ROCDL::RawPtrBufferStoreOp) { ++raw_stores; });
  EXPECT_EQ(raw_loads, 1);
  EXPECT_EQ(raw_stores, 1);
}

TEST(FlyDslCompilerTest, KeepsFourGiBAllocationsOnGlobalPointerPath) {
  mlir::DialectRegistry registry;
  RegisterDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::LLVM::LLVMDialect,
                  mlir::ROCDL::ROCDLDialect>();
  mlir::MLIRContext context(registry);
  constexpr llvm::StringLiteral kModule = R"mlir(
    module {
      llvm.func @kernel(
          %input: !llvm.ptr {llvm.dereferenceable = 4294967296 : index},
          %output: !llvm.ptr {llvm.dereferenceable = 4294967296 : index}) {
        %value = llvm.load %input : !llvm.ptr -> f32
        llvm.store %value, %output : f32, !llvm.ptr
        llvm.return
      }
    }
  )mlir";
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kModule, &context);
  ASSERT_TRUE(module);

  mlir::PassManager wrap_pm(&context);
  AddGenericMemoryPasses(wrap_pm);
  ASSERT_TRUE(mlir::succeeded(wrap_pm.run(*module)));

  int make_pointers = 0;
  int integer_to_pointers = 0;
  module->walk([&](mlir::fly::MakePtrOp) { ++make_pointers; });
  module->walk([&](mlir::fly::IntToPtrOp) { ++integer_to_pointers; });
  EXPECT_EQ(make_pointers, 0);
  EXPECT_EQ(integer_to_pointers, 2);
}

TEST(FlyDslCompilerTest, KeepsWideVectorsOnGlobalPointerPath) {
  mlir::DialectRegistry registry;
  RegisterDialects(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::LLVM::LLVMDialect,
                  mlir::ROCDL::ROCDLDialect>();
  mlir::MLIRContext context(registry);
  constexpr llvm::StringLiteral kModule = R"mlir(
    module {
      llvm.func @kernel(
          %input: !llvm.ptr {llvm.dereferenceable = 4096 : index},
          %output: !llvm.ptr {llvm.dereferenceable = 4096 : index}) {
        %value = llvm.load %input : !llvm.ptr -> vector<8xf32>
        llvm.store %value, %output : vector<8xf32>, !llvm.ptr
        llvm.return
      }
    }
  )mlir";
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kModule, &context);
  ASSERT_TRUE(module);

  mlir::PassManager wrap_pm(&context);
  AddGenericMemoryPasses(wrap_pm);
  ASSERT_TRUE(mlir::succeeded(wrap_pm.run(*module)));

  int make_pointers = 0;
  int integer_to_pointers = 0;
  module->walk([&](mlir::fly::MakePtrOp) { ++make_pointers; });
  module->walk([&](mlir::fly::IntToPtrOp) { ++integer_to_pointers; });
  EXPECT_EQ(make_pointers, 0);
  EXPECT_EQ(integer_to_pointers, 2);
}

}  // namespace
}  // namespace xla::gpu::flydsl
