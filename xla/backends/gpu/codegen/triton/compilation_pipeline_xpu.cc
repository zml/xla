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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
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
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "xla/backends/gpu/codegen/triton/transforms/passes.h"
#include "xla/stream_executor/sycl/oneapi_compute_capability.h"

namespace xla::gpu {
namespace {

namespace mt = ::mlir::triton;
namespace mti = ::mlir::triton::intel;
namespace mtgi = ::mlir::triton::gpu::intel;

class SetTritonXpuModuleAttrsPass
    : public mlir::PassWrapper<SetTritonXpuModuleAttrsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  explicit SetTritonXpuModuleAttrsPass(int num_warps)
      : num_warps_(num_warps) {}

  llvm::StringRef getArgument() const override {
    return "xla-set-triton-xpu-module-attrs";
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

absl::StatusOr<std::string> TargetArch(
    const stream_executor::OneAPIComputeCapability& cc) {
  if (cc.IsBMG()) {
    return "bmg";
  }
  if (cc.IsPVC()) {
    return "pvc";
  }
  if (cc.IsDG2()) {
    return "dg2";
  }
  return absl::FailedPreconditionError(
      absl::StrCat("Unsupported oneAPI architecture for Triton XPU: ",
                   cc.ToString()));
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
  // TODO: Re-enable once oneAPI Triton autotuning has a trusted reference path.
  // The current DPAS lowering can compile and launch on BMG but produces wrong
  // Llama logits, and SYCL autotuning does not yet have a non-Triton GEMM
  // backend to reject a unanimously-wrong Triton cluster.
  options.supportDPAS = false;
  // XLA currently routes Triton-generated LLVM through the existing SYCL
  // SPIR-V path. Intel's 2D block IO lowering emits image-handle SPIR-V that
  // is not accepted by that path yet, so leave it disabled for autotuning.
  options.support2DBlockIO = false;
  // The current LLVM SPIR-V backend aliases OpPredicatedStoreINTEL with
  // OpConvertHandleToImageINTEL in module analysis, which rejects the i1
  // predicate operand as an image handle. Use regular masked IO for now.
  options.supportPredicatedIO = false;
  options.supportPrefetch256Bytes = cc.IsBMG();
  options.support256bLoadStore = cc.IsBMG();
  return options;
}

void MakeTTIR(mlir::OpPassManager* pm) {
  pm->addPass(mlir::createInlinerPass());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mt::createTritonCombineOps());
  pm->addPass(mt::createTritonReorderBroadcast());
  pm->addPass(mlir::createCSEPass());
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
  pm->addPass(mt::gpu::createTritonGPUCoalesce());
  if (cc.IsBMG()) {
    pm->addPass(mtgi::createTritonIntelGPUWidenLoadStoreEncoding());
  }
  pm->addPass(mtgi::createTritonIntelGPURemoveLayoutConversions());
  pm->addPass(mtgi::createTritonIntelGPUAccelerateMatmul());
  pm->addPass(mtgi::createTritonIntelGPUMaterializeBlockPointer());
  pm->addPass(mtgi::createTritonIntelGPURemoveLayoutConversions());
  pm->addPass(mtgi::createTritonIntelGPUOptimizeDotOperands());
  pm->addPass(mtgi::createTritonIntelGPUHoistLayoutConversions());
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
  pm->addPass(mtgi::createTritonIntelGPUAnnotateCacheControl());
  pm->addPass(mtgi::createTritonIntelGPUReduceDataDuplication());
  pm->addPass(mt::gpu::createTritonGPUReorderInstructions());
  pm->addPass(mt::createTritonLoopAwareCSE());
  pm->addPass(mlir::createSCCPPass());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addPass(mlir::createSymbolDCEPass());
  pm->addPass(mti::createTritonIntelRemoveMasks());
}

void MakeLLIR(mlir::OpPassManager* pm, int num_warps) {
  pm->addPass(std::make_unique<SetTritonXpuModuleAttrsPass>(num_warps));
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

absl::Status CreateTritonXpuPipeline(
    mlir::OpPassManager* pm,
    const stream_executor::OneAPIComputeCapability& oneapi_cc,
    int threads_per_warp, int num_warps, int num_ctas, int num_stages) {
  absl::StatusOr<std::string> target_arch = TargetArch(oneapi_cc);
  if (!target_arch.ok()) {
    return target_arch.status();
  }
  threads_per_warp = std::max(threads_per_warp, 1);
  MakeTTIR(pm);
  MakeTTGIR(pm, oneapi_cc, threads_per_warp, num_warps, num_ctas, num_stages,
            *target_arch);
  MakeLLIR(pm, num_warps);
  return absl::OkStatus();
}

}  // namespace xla::gpu
