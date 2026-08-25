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

#include "xla/backends/gpu/codegen/fp8_block_gemv_fusion.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/backends/gpu/codegen/kernels/fp8_block_gemv_kernel.h"
#include "xla/backends/gpu/codegen/triton/fp8_block_gemv.h"
#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {

AsyncThunkSequence Fp8BlockGemvFusion::Emit(
    IrEmitterContext& ir_emitter_context,
    const HloFusionInstruction& fusion) const {
  std::optional<Fp8BlockGemvSpec> spec = MatchFp8BlockGemv(fusion);
  if (!spec.has_value()) {
    return AsyncThunkSequence(absl::FailedPreconditionError(absl::StrCat(
        fusion.name(),
        " carries the CUDA fp8 block gemv kind but the matcher declines it; "
        "only Fp8BlockGemvBackend::ApplyConfig should produce this kind.")));
  }
  if (spec->batch != 1) {
    return AsyncThunkSequence(absl::FailedPreconditionError(
        absl::StrCat(fusion.name(), ": the CUDA kernel is single-row only, got "
                                    "batch ",
                     spec->batch)));
  }

  const xla::xtile::BlockLevelFusionConfig& block_config =
      analysis_.fusion_backend_config().block_level_fusion_config();
  if (block_config.output_tiles().empty() ||
      block_config.output_tiles(0).sizes_size() < 2) {
    return AsyncThunkSequence(absl::FailedPreconditionError(
        absl::StrCat(fusion.name(), ": no output tile to read the row count "
                                    "off")));
  }
  const int64_t rows_per_block = block_config.output_tiles(0).sizes(1);
  const int num_warps = block_config.num_warps();
  if (num_warps <= 0 || rows_per_block % num_warps != 0) {
    return AsyncThunkSequence(absl::FailedPreconditionError(absl::StrCat(
        fusion.name(), ": ", rows_per_block, " rows do not divide over ",
        num_warps, " warps")));
  }
  kernel::Fp8BlockGemvKernelConfig config{
      /*num_warps=*/num_warps,
      /*rows_per_warp=*/static_cast<int>(rows_per_block / num_warps),
      /*unroll=*/block_config.num_stages()};

  ABSL_ASSIGN_OR_RETURN(emitters::KernelArguments kernel_arguments,
                   emitters::KernelArguments::Create(
                       ir_emitter_context.buffer_assignment(),
                       GetDefaultBufferAlignment(), &fusion));
  if (kernel_arguments.args().size() !=
      static_cast<size_t>(fusion.operand_count()) + 1) {
    return AsyncThunkSequence(absl::FailedPreconditionError(absl::StrCat(
        fusion.name(), ": expected ", fusion.operand_count() + 1,
        " kernel arguments, got ", kernel_arguments.args().size())));
  }

  absl::StatusOr<CustomKernel> custom_kernel = kernel::GetFp8BlockGemvKernel(
      config, spec->activation_index, spec->weight_index, spec->scale_index,
      /*output_arg=*/fusion.operand_count(), spec->n, spec->k);
  if (!custom_kernel.ok()) {
    return AsyncThunkSequence(custom_kernel.status());
  }

  return ThunkSequence::Of<CustomKernelThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          &fusion, ir_emitter_context.GetNextThunkId()),
      *std::move(custom_kernel), kernel_arguments);
}

}  // namespace xla::gpu
