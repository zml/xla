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
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "xla/backends/gpu/codegen/copy.h"
#include "xla/backends/gpu/codegen/cudnn.h"
#include "xla/backends/gpu/codegen/custom.h"
#include "xla/backends/gpu/codegen/emitters/concatenate.h"
#include "xla/backends/gpu/codegen/emitters/in_place_dynamic_update_slice.h"
#include "xla/backends/gpu/codegen/emitters/loop.h"
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
namespace {

class UnsupportedMusaFusion final : public FusionInterface {
 public:
  explicit UnsupportedMusaFusion(std::string error_message)
      : error_message_(std::move(error_message)) {}

  AsyncThunkSequence Emit(IrEmitterContext&,
                          const HloFusionInstruction&) const final {
    return absl::UnimplementedError(error_message_);
  }

 private:
  std::string error_message_;
};

std::unique_ptr<FusionInterface> CreateMlirKernelFusion(
    const HloFusionAnalysis& analysis,
    std::unique_ptr<MlirKernelEmitter> emitter) {
  if (analysis.device_info().gpu_compute_capability().IsMusa()) {
    return std::make_unique<MusaMlirKernelFusion>(std::move(emitter));
  }
  return std::make_unique<MlirKernelFusion>(std::move(emitter));
}

}  // namespace

Decision MusaFusionEmitterQualification(
    HloFusionAnalysis::EmitterFusionKind fusion_kind) {
  switch (fusion_kind) {
    case HloFusionAnalysis::EmitterFusionKind::kLoop:
    case HloFusionAnalysis::EmitterFusionKind::kReduction:
    case HloFusionAnalysis::EmitterFusionKind::kScatter:
    case HloFusionAnalysis::EmitterFusionKind::kTranspose:
    case HloFusionAnalysis::EmitterFusionKind::kConcatenate:
      return Decision::Allow();
    case HloFusionAnalysis::EmitterFusionKind::kCustomFusion:
      return Decision::Forbid(
          "custom fusion emission is not qualified for the MUSA backend");
    case HloFusionAnalysis::EmitterFusionKind::kTriton:
      return Decision::Forbid(
          "Triton fusion emission is not qualified for the MUSA backend");
    case HloFusionAnalysis::EmitterFusionKind::kCuDnn:
      return Decision::Forbid(
          "cuDNN fusion emission is not qualified for the MUSA backend");
    case HloFusionAnalysis::EmitterFusionKind::kSort:
      return Decision::Forbid(
          "sort fusion emission is not qualified for the MUSA backend");
  }
  return Decision::Forbid(
      "unknown fusion emission is not qualified for the MUSA backend");
}

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

std::unique_ptr<FusionInterface> GetFusionEmitter(
    const FusionInfo& fusion_info) {
  const auto& analysis = fusion_info.analysis();
  if (analysis.device_info().gpu_compute_capability().IsMusa()) {
    Decision qualification =
        MusaFusionEmitterQualification(analysis.emitter_fusion_kind());
    if (qualification.IsForbidden()) {
      return std::make_unique<UnsupportedMusaFusion>(qualification.Explain());
    }
  }
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
        return CreateMlirKernelFusion(
            analysis,
            std::make_unique<InPlaceDynamicUpdateSliceFusion>(analysis));
      }
      return CreateMlirKernelFusion(analysis,
                                    std::make_unique<LoopFusion>(analysis));
    }
    case HloFusionAnalysis::EmitterFusionKind::kReduction: {
      return CreateMlirKernelFusion(analysis, CreateReductionFusion(analysis));
    }
    case HloFusionAnalysis::EmitterFusionKind::kScatter: {
      return CreateMlirKernelFusion(analysis, CreateScatterFusion(analysis));
    }
    case HloFusionAnalysis::EmitterFusionKind::kTranspose: {
      return CreateMlirKernelFusion(analysis, CreateTransposeFusion(analysis));
    }
    case HloFusionAnalysis::EmitterFusionKind::kConcatenate: {
      return CreateMlirKernelFusion(
          analysis, std::make_unique<ConcatenateFusion>(analysis));
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
