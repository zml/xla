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

#include <algorithm>
#include <string>

#include "intel/include/Dialect/Triton/Transforms/Passes.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "intel/include/TritonAnnotateModule/Passes.h"
#include "intel/include/TritonGENToLLVM/Passes.h"
#include "intel/include/TritonIntelGPUToLLVM/Passes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "xla/stream_executor/sycl/oneapi_compute_capability.h"

namespace xla {
namespace gpu {
namespace {

namespace mt = ::mlir::triton;
namespace mti = ::mlir::triton::intel;
namespace mtgi = ::mlir::triton::gpu::intel;

class SetTritonOneApiModuleAttrsPass
    : public mlir::PassWrapper<SetTritonOneApiModuleAttrsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  explicit SetTritonOneApiModuleAttrsPass(int num_warps)
      : num_warps_(num_warps) {}

  llvm::StringRef getArgument() const override {
    return "xla-set-triton-oneapi-module-attrs";
  }

  llvm::StringRef getDescription() const override {
    return "Sets Triton module attributes required by the Intel XPU backend";
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::Builder builder(module.getContext());
    if (!module->hasAttr("ttg.total-num-warps")) {
      module->setAttr("ttg.total-num-warps",
                      builder.getI32IntegerAttr(num_warps_));
    }
  }

 private:
  int num_warps_;
};

std::string TargetArch(const stream_executor::OneAPIComputeCapability& cc) {
  if (cc.IsBMG()) {
    return "bmg";
  }
  if (cc.IsPVC()) {
    return "pvc";
  }
  if (cc.IsDG2()) {
    return "dg2";
  }
  return "spir64";
}

mtgi::TritonAnnotateModuleOptions AnnotationOptions(
    const stream_executor::OneAPIComputeCapability& cc, int threads_per_warp,
    const std::string& target_arch) {
  mtgi::TritonAnnotateModuleOptions options;
  options.minSGSize = std::max(threads_per_warp, 1);
  options.threadsPerWarp = options.minSGSize;
  options.targetArch = target_arch;

  const bool supports_bf16 = cc.IsBMG() || cc.IsPVC();
  options.supportBF16Conversion = supports_bf16;
  options.supportBfloat16Arithmetic = supports_bf16;
  options.supportDPAS = cc.IsBMG() || cc.IsPVC();
  // Keep these disabled until the existing LLVM SPIR-V path accepts the
  // corresponding Intel extension patterns end-to-end.
  options.support2DBlockIO = false;
  options.supportPredicatedIO = false;
  options.supportPrefetch256Bytes = cc.IsBMG();
  return options;
}

void MakeTTIR(mlir::OpPassManager* pm) {
  pm->addPass(mlir::createInlinerPass());
  pm->addPass(mti::createTritonIntelBlockPointerToTensorDesc());
  pm->addPass(mti::createTritonIntelTensorDescToBlockPointer());
  pm->addPass(mt::createTritonRewriteTensorDescriptorToPointer());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mti::createTritonIntelRemoveBoundaryChecks());
  pm->addPass(mti::createTritonIntelRemoveMasks());
  pm->addPass(mti::createTritonIntelStrideVersioning());
  pm->addPass(mti::createTritonIntelFuseReshape());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mt::createTritonCombineOps());
  pm->addPass(mt::createTritonReorderBroadcast());
  pm->addPass(mlir::createCSEPass());
  pm->addPass(mlir::createLoopInvariantCodeMotionPass());
  pm->addPass(mlir::createSymbolDCEPass());
  pm->addPass(mt::createTritonLoopUnroll());
}

void MakeTTGIR(mlir::OpPassManager* pm,
               const stream_executor::OneAPIComputeCapability& cc,
               int threads_per_warp, int num_warps, int num_ctas,
               int num_stages, const std::string& target_arch) {
  pm->addPass(mtgi::createTritonAnnotateModule(
      AnnotationOptions(cc, threads_per_warp, target_arch)));
  pm->addPass(mt::createConvertTritonToTritonGPU(
      {"xpu", num_warps, threads_per_warp, num_ctas}));
  pm->addPass(mtgi::createTritonIntelGPUCoalesce());
  pm->addPass(mtgi::createTritonIntelGPURemoveLayoutConversions());
  pm->addPass(mtgi::createTritonIntelGPUAccelerateMatmul());
  pm->addPass(mtgi::createTritonIntelGPUMaterializeBlockPointer());
  pm->addPass(mtgi::createTritonIntelGPURemoveLayoutConversions());
  pm->addPass(mtgi::createTritonIntelGPUOptimizeDotOperands());
  pm->addPass(mtgi::createTritonIntelGPUPipeline({num_stages, false}));
  pm->addPass(mtgi::createTritonIntelGPUReduceVariableLiveness());
  pm->addPass(mt::createTritonLoopAwareCSE());
  pm->addPass(mt::gpu::createTritonGPUFuseNestedLoops());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mt::createTritonLoopInvariantCodeMotion());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mt::gpu::createTritonGPUCombineTensorSelectAndIf());
  pm->addPass(mt::gpu::createTritonGPUOptimizeThreadLocality());
  pm->addPass(mtgi::createTritonIntelGPUOptimizeDotOperands());
  pm->addPass(mtgi::createTritonIntelGPUOptimizeReductionLocality());
  pm->addPass(mtgi::createTritonIntelGPUReduceDataDuplication());
  pm->addPass(mt::gpu::createTritonGPUReorderInstructions());
  pm->addPass(mt::createTritonLoopAwareCSE());
  pm->addPass(mlir::createSCCPPass());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::createSymbolDCEPass());
}

void MakeLLIR(mlir::OpPassManager* pm, int num_warps) {
  pm->addPass(std::make_unique<SetTritonOneApiModuleAttrsPass>(num_warps));
  pm->addPass(mlir::createSCFToControlFlowPass());
  pm->addPass(mlir::createInlinerPass());
  pm->addPass(mlir::createConvertIndexToLLVMPass());
  pm->addPass(mtgi::createIntelAllocateSharedMemory());
  pm->addPass(mt::gpu::createTritonGPUGlobalScratchAllocationPass());
  pm->addPass(mtgi::createConvertTritonIntelGPUToLLVM());
  pm->addPass(mt::createConvertTritonGENToLLVM());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mtgi::createTritonIntelGPURewriteStackPtr());
  pm->addPass(mlir::createCSEPass());
  pm->addPass(mlir::createConvertControlFlowToLLVMPass());
  pm->addPass(mlir::createArithToLLVMConversionPass());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::createCSEPass());
  pm->addPass(mlir::createSymbolDCEPass());
}

}  // namespace

void CreateTritonOneApiPipeline(
    mlir::OpPassManager* pm,
    const stream_executor::OneAPIComputeCapability& oneapi_cc, int num_warps,
    int num_ctas, int num_stages) {
  constexpr int kDpasSubgroupSize = 16;
  MakeTTIR(pm);
  MakeTTGIR(pm, oneapi_cc, kDpasSubgroupSize, num_warps, num_ctas, num_stages,
            TargetArch(oneapi_cc));
  MakeLLIR(pm, num_warps);
}

}  // namespace gpu
}  // namespace xla
