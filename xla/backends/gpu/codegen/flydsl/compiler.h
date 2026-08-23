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

// Marks an XLA-emitted module whose global memory boundary must be expressed
// through Fly before translation to LLVM IR.
void MarkGenericFusion(mlir::ModuleOp module);

// Returns true for modules marked by MarkGenericFusion.
bool IsGenericFusion(mlir::ModuleOp module);

// Re-expresses ordinary LLVM global loads and stores rooted at kernel
// arguments as fly.ptr.load/fly.ptr.store. This pass runs after XLA has
// lowered its tensor ABI and before the Fly-to-ROCDL conversion.
void AddGenericMemoryPasses(mlir::OpPassManager& pm);

// Lowers Fly IR to upstream MLIR and ROCDL. XLA's standard GPU pipeline is
// responsible for subsequent lowering to the LLVM dialect and LLVM IR.
void AddLoweringPasses(mlir::OpPassManager& pm,
                       bool restore_generic_memory_metadata = false);

}  // namespace xla::gpu::flydsl

#endif  // XLA_BACKENDS_GPU_CODEGEN_FLYDSL_COMPILER_H_
