/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/codegen/triton/compilation_pipeline.h"

#include "absl/log/log.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h"
#include "xla/backends/gpu/codegen/triton/transforms/passes.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

void CreateTritonXlaPipeline(
    mlir::OpPassManager* pm,
    const stream_executor::GpuComputeCapability& gpu_cc, bool rewrite_int4,
    bool allow_tma, int num_stages, bool warp_specialization_allowed,
    bool enable_pdl) {
  pm->addPass(mlir::triton::xla::createTritonXLASqueezeDimsPass());
  pm->addPass(mlir::triton::xla::createTritonXLAFoldTransposePass());
  pm->addPass(mlir::triton::xla::createTritonXLALowerBlockBarrierPass());
  pm->addPass(mlir::triton::xla::createTritonXLALowerAtomicsPass());
  pm->addPass(mlir::triton::xla::createTritonXLALowerGetTidPass());
  pm->addPass(mlir::triton::xla::createTritonXLALowerXTilePass());
  mlir::triton::xla::StableHLOLowerToTritonPassOptions stablehlo_raw_options;
  stablehlo_raw_options.warp_specialization_allowed_ =
      warp_specialization_allowed;
  pm->addPass(mlir::triton::xla::createStableHLOLowerToTritonPass(
      stablehlo_raw_options));
  pm->addPass(mlir::triton::xla::createTritonXLAFoldReshapeAroundForLoopPass());

  pm->addPass(emitters::createSafeIntegerArithmeticPass());
  pm->addPass(mlir::triton::xla::createUnsupportedElementwiseToTritonPass());

  auto* cuda_cc = gpu_cc.cuda_compute_capability();
  bool is_at_least_hopper = cuda_cc != nullptr && cuda_cc->IsAtLeastHopper();

  auto* rocm_cc = gpu_cc.rocm_compute_capability();
  bool rocm_supports_tdm = rocm_cc != nullptr && rocm_cc->has_tdm_support();

  if (rewrite_int4) {
    mlir::triton::xla::LoadInt4RewritePassOptions options;
    options.enable_bf16x2_ = is_at_least_hopper;
    pm->addPass(mlir::triton::xla::createLoadInt4RewritePass(options));
  }

  if (enable_pdl) {
    pm->addPass(createInsertPDLPass());
  }
  mlir::triton::xla::TritonXLAExtractInsertToTritonPassOptions
      extract_insert_options;
  extract_insert_options.allow_tma_ = allow_tma && is_at_least_hopper;
  extract_insert_options.allow_tdm_ = rocm_supports_tdm;
  extract_insert_options.num_stages_ = num_stages;
  pm->addPass(mlir::triton::xla::createTritonXLAExtractInsertToTritonPass(
      extract_insert_options));
  if (enable_pdl) {
    pm->addPass(emitters::createLowerPdlWaitPass());
  }

  // Lower affine expressions into arithmetic ops.
  pm->addPass(mlir::createLowerAffinePass());

  // Lower xla_gpu.apply_indexing into arithmetic ops.
  pm->addPass(emitters::createSimplifyAffinePass());
  pm->addPass(createConvertIndexTypePass());
  pm->addPass(mlir::createCompositeFixedPointPass(
      "TritonXLAUnswitchLoopsComposite", [](mlir::OpPassManager& pm) {
        // Loop unswitcher needs loop invariant code to be outside of the loop.
        pm.addPass(mlir::createLoopInvariantCodeMotionPass());
        pm.addPass(mlir::triton::xla::createTritonXLAUnswitchLoopsPass());
        pm.addPass(mlir::createCanonicalizerPass());
      }));
}

#if defined(GOOGLE_CUDA)
void CreateTritonCudaPipeline(
    mlir::OpPassManager* pm,
    const stream_executor::CudaComputeCapability& cuda_cc, int num_warps,
    int num_ctas, int num_stages);
#endif

#if defined(TENSORFLOW_USE_ROCM)
void CreateTritonRocmPipeline(
    mlir::OpPassManager* pm,
    const stream_executor::RocmComputeCapability& rocm_cc, int num_warps,
    int num_ctas, int num_stages);
#endif

#if defined(TENSORFLOW_USE_SYCL)
void CreateTritonOneApiPipeline(
    mlir::OpPassManager* pm,
    const stream_executor::OneAPIComputeCapability& oneapi_cc, int num_warps,
    int num_ctas, int num_stages);
#endif

void CreateTritonPipeline(mlir::OpPassManager* pm,
                          const stream_executor::GpuComputeCapability& gpu_cc,
                          int num_warps, int num_ctas, int num_stages) {
#if defined(GOOGLE_CUDA)
  if (auto* cuda_cc = gpu_cc.cuda_compute_capability()) {
    return CreateTritonCudaPipeline(pm, *cuda_cc, num_warps, num_ctas,
                                    num_stages);
  }
#endif
#if defined(TENSORFLOW_USE_SYCL)
  if (auto* oneapi_cc = gpu_cc.oneapi_compute_capability()) {
    return CreateTritonOneApiPipeline(pm, *oneapi_cc, num_warps, num_ctas,
                                      num_stages);
  }
#endif
#if defined(TENSORFLOW_USE_ROCM)
  if (auto* rocm_cc = gpu_cc.rocm_compute_capability()) {
    return CreateTritonRocmPipeline(pm, *rocm_cc, num_warps, num_ctas,
                                    num_stages);
  }
#endif

  LOG(FATAL) << "Unsupported GPU target for Triton: " << gpu_cc.ToString();
}

}  // namespace xla::gpu
