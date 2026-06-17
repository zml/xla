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
#include <cstdlib>
#include <string>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "intel/include/Dialect/Triton/Transforms/Passes.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "intel/include/TritonAnnotateModule/Passes.h"
#include "intel/include/TritonGENToLLVM/Passes.h"
#include "intel/include/TritonIntelGPUToLLVM/Passes.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
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

// Mirrors XPUBackend.annotate_module() as closely as possible
mtgi::TritonAnnotateModuleOptions AnnotationOptions(
    const stream_executor::OneAPIComputeCapability& cc, int threads_per_warp,
    const std::string& target_arch) {
  mtgi::TritonAnnotateModuleOptions options;
  const bool is_pvc_or_bmg = cc.IsBMG() || cc.IsPVC();

  options.minSGSize = std::max(threads_per_warp, 1);
  options.threadsPerWarp = options.minSGSize;
  options.targetArch = target_arch;

  options.supportDPAS = is_pvc_or_bmg;
  options.supportBF16Conversion = is_pvc_or_bmg;      // has_bfloat16_conversion
  options.supportBfloat16Arithmetic = is_pvc_or_bmg;  // has_bfloat16_arithmetic

  options.support2DBlockIO = is_pvc_or_bmg;
  options.supportPredicatedIO = is_pvc_or_bmg;

  // has_256b_prefetch / has_256b_load_store are an Xe3P+ feature; the python
  // backend defaults both to False (parse_target, pending a driver prop) and
  // PVC/BMG are Xe2, so they are off here (matching python).
  // (OneAPIComputeCapability exposes no Xe3P / 256b capability query.)
  options.supportPrefetch256Bytes = false;
  options.support256bLoadStore = false;

  options.supportRoundedDivideSqrt = is_pvc_or_bmg;
  options.useClRoundedDivideSqrt = false;
  options.isLTS = false;
  options.isFastMath = false;
  return options;
}

// Mirrors XPUBackend.make_ttir().
void MakeTTIR(mlir::OpPassManager* pm) {
  pm->addPass(mlir::createInlinerPass());                            // inliner
  pm->addPass(mti::createTritonRewriteTensorDescriptorToPointer());  // rewrite_tensor_descriptor_to_pointer
  pm->addPass(mlir::createCSEPass());                                // cse
  pm->addPass(mt::createTritonLoopInvariantCodeMotion());            // triton_licm
  pm->addPass(mti::createTritonIntelRemoveMasks());                  // remove_masks
  pm->addPass(mti::createTritonIntelStrideVersioning());             // stride_versioning
  pm->addPass(mti::createTritonIntelFuseReshape());                  // fuse_reshape
  pm->addPass(mti::createTritonIntelGPUFoldTrueCmpI());              // fold_true_cmpi
  pm->addNestedPass<mt::FuncOp>(
      mti::createTritonIntelGPUPrepareIfCombining());                // prepare_if_combining (FuncOp pass)
  pm->addPass(mlir::createCanonicalizerPass());                      // canonicalizer
  pm->addPass(mt::createTritonCombineOps());                         // combine
  pm->addPass(mti::createTritonIntelSimplifySignedArithmetic());     // simplify_signed_arithmetic
  pm->addPass(mt::createTritonReorderBroadcast());                   // reorder_broadcast
  pm->addPass(mlir::createCSEPass());                                // cse
  pm->addPass(mlir::createSymbolDCEPass());                          // symbol_dce
  pm->addPass(mt::createTritonLoopUnroll());                         // loop_unroll
}

// Mirrors XPUBackend.make_ttgir(). Omits only env-gated/instrumentation passes
// that are off by default (TRITON_INTEL_ANNOTATE_LATENCIES, fpsan).
void MakeTTGIR(mlir::OpPassManager* pm,
               const stream_executor::OneAPIComputeCapability& cc,
               int threads_per_warp, int num_warps, int num_ctas,
               int num_stages, const std::string& target_arch) {
  // annotate_module runs in its own pass manager in python; for DPAS targets
  // (threads_per_warp == min_sg_size == 16) the warp_size readback is a no-op,
  // so streaming both passes into one manager is equivalent here.
  pm->addPass(mtgi::createTritonAnnotateModule(
      AnnotationOptions(cc, threads_per_warp, target_arch)));
  pm->addPass(mt::createConvertTritonToTritonGPU(
      {"xpu", num_warps, threads_per_warp, num_ctas}));  // convert_to_ttgpuir
  pm->addPass(mtgi::createTritonIntelGPUCoalesce());     // coalesce
  // widen_load_store_encoding is gated in python on has_256b_load_store, an
  // Xe3P+ feature that is False on Xe2 PVC/BMG, so it is not run here.
  pm->addPass(mtgi::createTritonIntelGPURemoveLayoutConversions());  // remove_layout_conversions
  pm->addPass(mtgi::createTritonIntelGPUAccelerateMatmul());         // accelerate_matmul
  pm->addPass(mtgi::createTritonIntelGPUMaterializeBlockPointer());  // materialize_block_pointer
  pm->addPass(mtgi::createTritonIntelGPURemoveLayoutConversions());  // remove_layout_conversions
  pm->addPass(mtgi::createTritonIntelGPUOptimizeDotOperands());      // optimize_dot_operands (intel)
  pm->addPass(mtgi::createTritonIntelGPUHoistLayoutConversions(
      {/*grfMode=*/"default"}));                          // hoist_layout_conversions
  pm->addPass(mtgi::createTritonIntelGPUPipeline(
      {num_stages, /*useBarrier=*/false}));               // pipeline
  pm->addPass(mtgi::createTritonIntelGPUReduceVariableLiveness());  // reduce_variable_liveness
  pm->addPass(mt::createTritonLoopAwareCSE());           // loop_aware_cse
  pm->addPass(mt::gpu::createTritonGPUFuseNestedLoops());  // fuse_nested_loops
  pm->addPass(mlir::createCanonicalizerPass());          // canonicalizer
  pm->addPass(mt::createTritonLoopInvariantCodeMotion());  // triton_licm
  pm->addPass(mlir::createCanonicalizerPass());          // canonicalizer
  pm->addPass(mt::gpu::createTritonGPUCombineTensorSelectAndIf());  // combine_tensor_select_and_if
  pm->addPass(mt::gpu::createTritonGPUOptimizeThreadLocality());    // optimize_thread_locality
  pm->addPass(mt::gpu::createTritonGPUOptimizeDotOperands(
      {/*hoistLayoutConversion=*/true}));                 // optimize_dot_operands (upstream)
  pm->addPass(mlir::createCSEPass());                    // cse
  pm->addPass(mt::gpu::createTritonGPUPrefetch());       // prefetch
  pm->addPass(mt::gpu::createTritonGPUOptimizeDotOperands(
      {/*hoistLayoutConversion=*/true}));                 // optimize_dot_operands (upstream)
  pm->addPass(mtgi::createTritonIntelGPURemoveLayoutConversions());  // remove_layout_conversions
  pm->addPass(mtgi::createTritonIntelGPUAnnotateCacheControl());     // annotate_cache_control
  pm->addPass(mtgi::createTritonIntelGPUReduceDataDuplication());    // reduce_data_duplication
  pm->addPass(mt::gpu::createTritonGPUReorderInstructions());        // reorder_instructions
  pm->addPass(mt::createTritonLoopAwareCSE());           // loop_aware_cse
  pm->addPass(mlir::createSymbolDCEPass());              // symbol_dce
  pm->addPass(mlir::createSCCPPass());                   // sccp
  pm->addPass(mlir::createCanonicalizerPass());          // canonicalizer

  if (std::getenv("TRITON_INTEL_OPTIMIZE_REDUCTION_LOCALITY") != nullptr){
    pm->addPass(mtgi::createTritonIntelGPUOptimizeReductionLocality());  // optimize_reduction_locality
  }
  
  // arith_emulate_unsupported_floats(["bf16"], "f32").
  mlir::arith::ArithEmulateUnsupportedFloatsOptions emulate_opts;
  emulate_opts.sourceTypeStrs = {"bf16"};
  emulate_opts.targetTypeStr = "f32";
  pm->addPass(mlir::arith::createArithEmulateUnsupportedFloats(emulate_opts));
}

// Mirrors XPUBackend.make_llir()
void MakeLLIR(mlir::OpPassManager* pm, int num_warps) {
  pm->addPass(std::make_unique<SetTritonOneApiModuleAttrsPass>(num_warps));
  pm->addPass(mtgi::createTritonIntelGPULowerTo2DBlockLoad());  // lower_to_2d_block_load
  pm->addPass(mlir::createSCFToControlFlowPass());              // scf_to_cf
  pm->addPass(mlir::createInlinerPass());                       // inliner (gluon)
  pm->addPass(mlir::createConvertIndexToLLVMPass());            // index_to_llvmir
  pm->addPass(mtgi::createIntelAllocateSharedMemory());         // allocate_shared_memory
  pm->addPass(mt::gpu::createTritonGPUGlobalScratchAllocationPass());  // allocate_global_scratch_memory
  pm->addPass(mtgi::createConvertTritonIntelGPUToLLVM());       // to_llvmir
  pm->addPass(mt::createConvertTritonGENToLLVM());              // gen_to_llvm
  pm->addPass(mlir::createCanonicalizerPass());                 // canonicalizer
  pm->addPass(mtgi::createTritonIntelGPURewriteStackPtr());     // rewrite_stack_ptr
  pm->addPass(mlir::createCSEPass());                           // cse
  pm->addPass(mlir::createConvertControlFlowToLLVMPass());      // (XLA lowering glue)
  pm->addPass(mlir::createArithToLLVMConversionPass());         // arith_to_llvmir
  pm->addPass(mlir::createCanonicalizerPass());                 // canonicalizer
  pm->addPass(mlir::createCSEPass());                           // cse
  pm->addPass(mlir::createSymbolDCEPass());                     // symbol_dce
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
