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
#ifndef XLA_BACKENDS_GPU_CODEGEN_FLYDSL_COMPILER_H_
#define XLA_BACKENDS_GPU_CODEGEN_FLYDSL_COMPILER_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"

namespace xla::gpu::flydsl {

// Registers only the native MLIR dialects required to parse and lower Fly IR.
void RegisterDialects(mlir::DialectRegistry& registry);

// Returns true if module contains operations owned by FlyDSL dialects.
bool HasOperations(mlir::ModuleOp module);

// Lowers Fly IR to upstream MLIR and ROCDL. XLA's standard GPU pipeline is
// responsible for subsequent lowering to the LLVM dialect and LLVM IR.
void AddLoweringPasses(mlir::OpPassManager& pm);

}  // namespace xla::gpu::flydsl

#endif  // XLA_BACKENDS_GPU_CODEGEN_FLYDSL_COMPILER_H_
