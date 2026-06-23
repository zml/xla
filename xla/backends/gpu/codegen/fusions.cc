/* Copyright 2023 The OpenXLA Authors.

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
#include "xla/backends/gpu/codegen/fusions.h"

#include <memory>
#include <optional>
#include <utility>

#include "xla/backends/gpu/codegen/copy.h"
#include "xla/backends/gpu/codegen/cudnn.h"
#include "xla/backends/gpu/codegen/custom.h"
#include "xla/backends/gpu/codegen/emitters/concatenate.h"
#include "xla/backends/gpu/codegen/emitters/in_place_dynamic_update_slice.h"
#include "xla/backends/gpu/codegen/emitters/loop.h"
#if defined(TENSORFLOW_USE_METAL)
#include "xla/backends/gpu/codegen/emitters/metal_mlir_kernel_fusion.h"
#endif
#include "xla/backends/gpu/codegen/emitters/mlir_kernel_emitter.h"
#include "xla/backends/gpu/codegen/emitters/reduction.h"
#include "xla/backends/gpu/codegen/emitters/scatter.h"
#include "xla/backends/gpu/codegen/emitters/transpose.h"
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/backends/gpu/codegen/sort.h"
#include "xla/backends/gpu/codegen/triton/fusion.h"
#include "xla/codegen/ir_emission_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/layout_util.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape.h"

namespace xla {
namespace gpu {

std::optional<std::unique_ptr<FusionInterface>> HloFusionInfo::GetCopyFusion()
    const {
  for (const HloInstructionAdaptor& root_adaptor : analysis().fusion_roots()) {
    const HloInstruction* root = &root_adaptor.instruction();
    if (root->opcode() != HloOpcode::kCopy ||
        root->operand(0)->opcode() != HloOpcode::kParameter ||
        !LayoutUtil::Equal(root->operand(0)->shape().layout(),
                           root->shape().layout())) {
      return std::nullopt;
    }
  }

  return std::make_unique<MemcpyFusion>(analysis());
}

bool HloFusionInfo::CanEmitDynamicUpdateSliceInPlace() const {
  auto ret = CanEmitFusedDynamicUpdateSliceInPlace(analysis().fusion(),
                                                   buffer_assignment_, instr_);
  return ret.ok() && *ret;
}

// Wraps an MlirKernelEmitter in the fusion type used for the current platform.
// On the macOS/Metal build (where GetFusionEmitter is only ever reached for
// Metal fusions — no CUDA/ROCm on macOS, and the legacy Metal path doesn't use
// this) it returns a MetalMlirKernelFusion, which overrides only CreateLLVMModule
// to emit Apple AIR instead of lowering MLIR->NVVM, and fails loud on any fused
// DAG the AIR emitter can't lower (never a silently-wrong NVVM kernel). It
// subclasses MlirKernelFusion, so the cost model's dynamic_cast and the inherited
// launch_dimensions()/indexing still resolve. Elsewhere it is a plain
// MlirKernelFusion (no behavior change).
static std::unique_ptr<KernelFusionInterface> MakeMlirFusion(
    std::unique_ptr<MlirKernelEmitter> emitter) {
#if defined(TENSORFLOW_USE_METAL)
  return std::make_unique<MetalMlirKernelFusion>(std::move(emitter));
#else
  return std::make_unique<MlirKernelFusion>(std::move(emitter));
#endif
}

std::unique_ptr<FusionInterface> GetFusionEmitter(
    const FusionInfo& fusion_info) {
  const auto& analysis = fusion_info.analysis();
  switch (analysis.emitter_fusion_kind()) {
    case HloFusionAnalysis::EmitterFusionKind::kCustomFusion:
      return std::make_unique<CustomFusion>();
    case HloFusionAnalysis::EmitterFusionKind::kLoop: {
      // Check for a memcpy fusion before checking if a DUS can be emitted in
      // place.
      if (auto copy_fusion = fusion_info.GetCopyFusion()) {
        return *std::move(copy_fusion);
      }
      if (IsDynamicUpdateSliceFusion(analysis.fusion_spec()) &&
          fusion_info.CanEmitDynamicUpdateSliceInPlace()) {
        return MakeMlirFusion(
            std::make_unique<InPlaceDynamicUpdateSliceFusion>(analysis));
      }
      return MakeMlirFusion(std::make_unique<LoopFusion>(analysis));
    }
    case HloFusionAnalysis::EmitterFusionKind::kReduction: {
      return MakeMlirFusion(CreateReductionFusion(analysis));
    }
    case HloFusionAnalysis::EmitterFusionKind::kScatter: {
      return MakeMlirFusion(CreateScatterFusion(analysis));
    }
    case HloFusionAnalysis::EmitterFusionKind::kTranspose: {
      return MakeMlirFusion(CreateTransposeFusion(analysis));
    }
    case HloFusionAnalysis::EmitterFusionKind::kConcatenate: {
      return MakeMlirFusion(std::make_unique<ConcatenateFusion>(analysis));
    }
    case HloFusionAnalysis::EmitterFusionKind::kSort: {
      return std::make_unique<SortFusion>();
    }
    case HloFusionAnalysis::EmitterFusionKind::kTriton:
      return std::make_unique<TritonFusion>(analysis);
    case HloFusionAnalysis::EmitterFusionKind::kCuDnn:
      return std::make_unique<CuDnnFusion>(analysis);
  }
}

}  // namespace gpu
}  // namespace xla
