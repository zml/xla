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

#ifndef XLA_CODEGEN_EMITTERS_TRANSFORMS_MUSA_GPU_TO_LLVM_H_
#define XLA_CODEGEN_EMITTERS_TRANSFORMS_MUSA_GPU_TO_LLVM_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace xla::emitters {

inline constexpr char kMusaKernelMarker[] = "xla.musa.kernel.v1";

// Installs the qualified MUSA target contract before any pointer- or
// alignment-sensitive lowering runs.
mlir::LogicalResult ConfigureMusaLLVMModule(mlir::ModuleOp module);

// Lowers only the GPU operations qualified by MUSA shim mapping version 1.
// The GPU dialect is illegal after this conversion, so every other GPU op
// fails closed instead of falling through to NVVM or ROCDL.
void PopulateMusaGpuToLLVMConversionPatterns(
    const mlir::LLVMTypeConverter& converter, mlir::RewritePatternSet& patterns,
    mlir::ConversionTarget& target);

}  // namespace xla::emitters

#endif  // XLA_CODEGEN_EMITTERS_TRANSFORMS_MUSA_GPU_TO_LLVM_H_
