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

#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "xla/backends/gpu/codegen/triton/compilation_pipeline.h"
#include "xla/debug_options_flags.h"
#include "xla/pjrt/triton.h"
#include "xla/service/gpu/llvm_gpu_backend/spirv_backend.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/sycl/oneapi_compute_capability.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/tsl/platform/statusor.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace xla::triton {
namespace {

namespace se = ::stream_executor;

absl::StatusOr<se::OneAPIComputeCapability> ParseOneApiComputeCapability(
    absl::string_view arch_name) {
  se::OneAPIComputeCapability oneapi_cc(arch_name);
  if (oneapi_cc.generation() == 0 && oneapi_cc.version() == 0) {
    return absl::InvalidArgumentError(
        "Unknown OneAPI Triton target architecture");
  }
  return oneapi_cc;
}

int64_t SharedMemoryBytes(mlir::ModuleOp module) {
  if (auto attr = module->getAttrOfType<mlir::IntegerAttr>("ttg.shared")) {
    return attr.getInt();
  }
  return 0;
}

int32_t GlobalScratchSize(mlir::ModuleOp module) {
  if (auto attr = module->getAttrOfType<mlir::IntegerAttr>(
          "ttg.global_scratch_memory_size")) {
    return attr.getInt();
  }
  return 0;
}

absl::StatusOr<std::string> LLVMToSPIRV(
    mlir::ModuleOp module, const se::OneAPIComputeCapability& oneapi_cc) {
  mlir::DialectRegistry registry;
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  module.getContext()->appendDialectRegistry(registry);

  llvm::LLVMContext llvm_context;
  std::unique_ptr<llvm::Module> llvm_module =
      mlir::translateModuleToLLVMIR(module, llvm_context);
  if (!llvm_module) {
    return absl::InternalError("Failed to emit LLVM IR");
  }

  llvm_module->setTargetTriple(llvm::Triple("spirv64-unknown-unknown"));
  for (llvm::Function& func : *llvm_module) {
    if (!func.isDeclaration() && func.hasExternalLinkage()) {
      func.setCallingConv(llvm::CallingConv::SPIR_KERNEL);
    }
  }

  DebugOptions debug_options = DefaultDebugOptionsIgnoringFlags();
  return gpu::spirv::CompileToSPIRV(
      llvm_module.get(), se::GpuComputeCapability(oneapi_cc), debug_options);
}

}  // namespace

absl::StatusOr<CompilationResult> Compile(absl::string_view module,
                                          absl::string_view arch_name,
                                          int num_warps, int num_ctas,
                                          int num_stages) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::triton::TritonDialect,
                      mlir::triton::gpu::TritonGPUDialect,
                      mlir::triton::gpu::intel::TritonIntelGPUDialect,
                      mlir::triton::TritonGEN::TritonGENDialect,
                      mlir::arith::ArithDialect, mlir::func::FuncDialect,
                      mlir::LLVM::LLVMDialect, mlir::spirv::SPIRVDialect,
                      mlir::tensor::TensorDialect>();
  mlir::DialectRegistry registry;
  mlir::func::registerInlinerExtension(registry);
  mlir::LLVM::registerInlinerInterface(registry);
  context.appendDialectRegistry(registry);

  mlir::OwningOpRef<mlir::ModuleOp> module_op =
      mlir::parseSourceString<mlir::ModuleOp>(module, &context);
  if (!module_op) {
    return absl::InvalidArgumentError("Failed to parse Triton module");
  }

  TF_ASSIGN_OR_RETURN(se::OneAPIComputeCapability oneapi_cc,
                      ParseOneApiComputeCapability(arch_name));

  mlir::PassManager pm(&context);
  pm.enableVerifier();
  gpu::CreateTritonPipeline(&pm, se::GpuComputeCapability(oneapi_cc), num_warps,
                            num_ctas, num_stages);
  if (failed(pm.run(*module_op))) {
    return absl::InternalError("Failed to compile Triton IR to LLVM IR");
  }

  TF_ASSIGN_OR_RETURN(auto spirv, LLVMToSPIRV(*module_op, oneapi_cc));

  return CompilationResult{
      SpirvBinary{spirv},
      SharedMemoryBytes(*module_op),
      GlobalScratchSize(*module_op),
  };
}

}  // namespace xla::triton
