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

#include "xla/service/gpu/metal_gpu_compiler.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout_util.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/metal_custom_call_executable.h"
#include "xla/service/gpu/metal_gpu_executable.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/metal/metal_platform_id.h"

namespace xla {
namespace gpu {

namespace {

absl::Status DirectAirNotWired() {
  return absl::UnimplementedError(
      "Metal direct AIR compilation is not wired into XLA yet.");
}

bool IsPackedSubbyteArray(const Shape& shape) {
  return shape.IsArray() &&
         (shape.element_type() == S4 || shape.element_type() == U4);
}

void SetPackedSubbyteLayouts(Shape* shape) {
  if (shape->IsTuple()) {
    for (Shape& subshape : *shape->mutable_tuple_shapes()) {
      SetPackedSubbyteLayouts(&subshape);
    }
    return;
  }
  if (IsPackedSubbyteArray(*shape)) {
    shape->mutable_layout()->set_element_size_in_bits(4);
  }
}

absl::Status SetPackedSubbyteLayouts(ShapeLayout* shape_layout) {
  Shape shape = shape_layout->shape();
  SetPackedSubbyteLayouts(&shape);
  return shape_layout->CopyLayoutFromShape(shape);
}

absl::Status NormalizePackedSubbyteLayouts(HloModule* module) {
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      SetPackedSubbyteLayouts(instruction->mutable_shape());
    }
  }

  ComputationLayout* entry_layout =
      module->mutable_entry_computation_layout();
  for (int64_t i = 0; i < entry_layout->parameter_count(); ++i) {
    TF_RETURN_IF_ERROR(
        SetPackedSubbyteLayouts(entry_layout->mutable_parameter_layout(i)));
  }
  TF_RETURN_IF_ERROR(
      SetPackedSubbyteLayouts(entry_layout->mutable_result_layout()));
  return absl::OkStatus();
}

absl::Status NormalizeRootLayoutToEntryOutput(HloModule* module) {
  HloInstruction* root = module->entry_computation()->root_instruction();
  const Shape& output_shape = module->entry_computation_layout().result_shape();
  if (!ShapeUtil::Compatible(root->shape(), output_shape)) {
    return absl::InvalidArgumentError(
        "Metal direct AIR root shape is not compatible with the entry output "
        "layout.");
  }
  return LayoutUtil::CopyLayoutBetweenShapes(output_shape,
                                             root->mutable_shape());
}

}  // namespace

se::Platform::Id MetalGpuCompiler::PlatformId() const {
  return stream_executor::metal::kMetalPlatformId;
}

absl::StatusOr<std::unique_ptr<HloModule>> MetalGpuCompiler::RunHloPasses(
    std::unique_ptr<HloModule> module, se::StreamExecutor* executor,
    const CompileOptions& options) {
  TF_RETURN_IF_ERROR(NormalizePackedSubbyteLayouts(module.get()));
  TF_RETURN_IF_ERROR(NormalizeRootLayoutToEntryOutput(module.get()));
  return std::move(module);
}

absl::StatusOr<std::unique_ptr<Executable>> MetalGpuCompiler::RunBackend(
    std::unique_ptr<HloModule> module, se::StreamExecutor* executor,
    const CompileOptions& options) {
  if (module->entry_computation()->root_instruction()->opcode() ==
      HloOpcode::kCustomCall) {
    auto shared_module = std::shared_ptr<HloModule>(std::move(module));
    return BuildMetalCustomCallExecutable(std::move(shared_module));
  }

  absl::StatusOr<MetalMatmulConfig> matmul_config = MatchMetalMatmul(*module);
  if (matmul_config.ok()) {
    TF_ASSIGN_OR_RETURN(std::vector<uint8_t> metallib,
                        CompileMetalMatmulAirToMetallib(*matmul_config));
    auto shared_module = std::shared_ptr<HloModule>(std::move(module));
    return std::make_unique<MetalMatmulExecutable>(
        std::move(shared_module), *matmul_config, std::move(metallib));
  }

  auto shared_module = std::shared_ptr<HloModule>(std::move(module));
  absl::StatusOr<std::unique_ptr<Executable>> reduction =
      BuildMetalReductionExecutable(shared_module);
  if (reduction.ok()) {
    return std::move(*reduction);
  }
  if (!absl::IsUnimplemented(reduction.status())) {
    return reduction.status();
  }

  absl::StatusOr<std::unique_ptr<Executable>> convert =
      BuildMetalConvertExecutable(shared_module);
  if (convert.ok()) {
    return std::move(*convert);
  }
  if (!absl::IsUnimplemented(convert.status())) {
    return convert.status();
  }

  absl::StatusOr<std::unique_ptr<Executable>> topk_tuple =
      BuildMetalTopKTupleExecutable(shared_module);
  if (topk_tuple.ok()) {
    return std::move(*topk_tuple);
  }
  if (!absl::IsUnimplemented(topk_tuple.status())) {
    return topk_tuple.status();
  }

  return BuildMetalElementwiseExecutable(std::move(shared_module));
}

absl::StatusOr<std::vector<std::unique_ptr<Executable>>>
MetalGpuCompiler::Compile(std::unique_ptr<HloModule> hlo_module,
                          std::vector<se::StreamExecutor*> stream_exec,
                          const CompileOptions& options) {
  if (stream_exec.empty()) {
    return absl::InvalidArgumentError(
        "Metal compilation requires at least one StreamExecutor.");
  }
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      RunHloPasses(std::move(hlo_module), stream_exec[0],
                                   options));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<Executable> executable,
                      RunBackend(std::move(module), stream_exec[0], options));
  std::vector<std::unique_ptr<Executable>> executables;
  executables.push_back(std::move(executable));
  return executables;
}

absl::StatusOr<std::vector<std::unique_ptr<CompiledModule>>>
MetalGpuCompiler::CompileAheadOfTime(std::unique_ptr<HloModule> module,
                                     const AotCompilationOptions& options) {
  return DirectAirNotWired();
}

HloCostAnalysis::ShapeSizeFunction
MetalGpuCompiler::ShapeSizeBytesFunction() const {
  return [](const Shape& shape) { return ShapeUtil::ByteSizeOf(shape, 8); };
}

}  // namespace gpu
}  // namespace xla
