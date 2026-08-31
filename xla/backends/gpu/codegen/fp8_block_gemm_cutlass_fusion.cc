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

#include "xla/backends/gpu/codegen/fp8_block_gemm_cutlass_fusion.h"

#include <cstddef>
#include <cstdint>
#include <optional>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "xla/backends/gpu/codegen/kernels/fp8_block_gemm_cutlass.h"
#include "xla/backends/gpu/codegen/triton/fp8_block_gemv.h"
#include "xla/backends/gpu/runtime/fp8_block_gemm_cutlass_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {

AsyncThunkSequence Fp8BlockGemmCutlassFusion::Emit(
    IrEmitterContext& ir_emitter_context,
    const HloFusionInstruction& fusion) const {
  std::optional<Fp8BlockGemvSpec> spec = MatchFp8BlockGemv(fusion);
  if (!spec.has_value()) {
    return AsyncThunkSequence(absl::FailedPreconditionError(absl::StrCat(
        fusion.name(),
        " carries the CUTLASS block gemm kind but the matcher declines it; "
        "only Fp8BlockGemvBackend::ApplyConfig should produce this kind.")));
  }
  // The collective reads a real per-row activation scale; a W8A16 fusion has an
  // identity scale that is not a buffer the kernel can walk.
  if (!spec->w8a8) {
    return AsyncThunkSequence(absl::FailedPreconditionError(
        absl::StrCat(fusion.name(),
                     ": the CUTLASS block gemm serves the W8A8 form only")));
  }

  const int config =
      analysis_.fusion_backend_config().fp8_block_gemm_cutlass_config()
          .config_index();
  if (config < 0 || config >= kernel::Fp8BlockGemmCutlassNumConfigs()) {
    return AsyncThunkSequence(absl::FailedPreconditionError(
        absl::StrCat(fusion.name(), ": config index ", config,
                     " is outside the compiled table")));
  }
  if (!kernel::Fp8BlockGemmCutlassCanRun(config, spec->batch, spec->n,
                                         spec->k)) {
    return AsyncThunkSequence(absl::FailedPreconditionError(absl::StrCat(
        fusion.name(), ": config ",
        kernel::Fp8BlockGemmCutlassConfigName(config), " cannot run m=",
        spec->batch, " n=", spec->n, " k=", spec->k)));
  }

  ABSL_ASSIGN_OR_RETURN(emitters::KernelArguments kernel_arguments,
                        emitters::KernelArguments::Create(
                            ir_emitter_context.buffer_assignment(),
                            GetDefaultBufferAlignment(), &fusion));
  const size_t expected = static_cast<size_t>(fusion.operand_count()) + 1;
  if (kernel_arguments.args().size() != expected) {
    return AsyncThunkSequence(absl::FailedPreconditionError(absl::StrCat(
        fusion.name(), ": expected ", expected, " kernel arguments, got ",
        kernel_arguments.args().size())));
  }

  const auto& args = kernel_arguments.args();
  return ThunkSequence::Of<Fp8BlockGemmCutlassThunk>(
      Thunk::ThunkInfo::WithProfileAnnotation(
          &fusion, ir_emitter_context.GetNextThunkId()),
      config, args[spec->activation_index].slice(),
      args[spec->act_scale_index].slice(), args[spec->weight_index].slice(),
      args[spec->scale_index].slice(), args[fusion.operand_count()].slice(),
      spec->batch, spec->n, spec->k);
}

}  // namespace xla::gpu
