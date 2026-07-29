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

#include "flydsl/Conversion/FlyToROCDL/FlyToROCDL.h"
#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Transforms/Passes.h"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/Passes.h"

namespace xla::gpu::flydsl {

void RegisterDialects(mlir::DialectRegistry& registry) {
  registry.insert<mlir::fly::FlyDialect, mlir::fly_rocdl::FlyROCDLDialect,
                  mlir::ub::UBDialect>();
}

bool HasOperations(mlir::ModuleOp module) {
  bool found = false;
  module->walk([&](mlir::Operation* op) {
    llvm::StringRef dialect = op->getName().getDialectNamespace();
    if (dialect == "fly" || dialect == "fly_rocdl") {
      found = true;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return found;
}

void AddLoweringPasses(mlir::OpPassManager& pm) {
  pm.addPass(mlir::fly::createFlyRewriteFuncSignaturePass());
  pm.addPass(mlir::fly::createFlyCanonicalizePass());
  pm.addPass(mlir::fly::createFlyLayoutLoweringPass());
  pm.addPass(mlir::fly::createFlyIntSwizzleSimplifyPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::fly::createFlyConvertAtomCallToSSAFormPass());
  pm.addPass(mlir::fly::createFlyPromoteRegMemToVectorSSAPass());
  pm.addPass(mlir::createFlyToROCDLConversionPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

}  // namespace xla::gpu::flydsl
