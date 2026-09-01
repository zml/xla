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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TRITON_FP8_BLOCK_GEMV_H_
#define XLA_BACKENDS_GPU_CODEGEN_TRITON_FP8_BLOCK_GEMV_H_

#include <cstdint>
#include <optional>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/codegen/xtile/block_level_parameters.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

struct Fp8BlockGemvSpec {
  int64_t activation_index;
  int64_t weight_index;
  int64_t scale_index;
  // Only meaningful when w8a8: the activation's own [batch, k/128] scale.
  int64_t act_scale_index;
  int64_t n;
  int64_t k;
  int64_t batch;
  bool activation_batch_major;
  // f8e4m3fn activation with a real per-row, per-128 scale (W8A8); otherwise
  // a bf16 activation with an identity scale (W8A16).
  bool w8a8;
  // The weight scale's element type. The matcher takes bf16, f32 and ue8m0,
  // but the hand-written CUDA kernel reads the buffer as bf16, so that rung
  // has to check this rather than assume it.
  PrimitiveType weight_scale_type;
};

inline constexpr absl::string_view kFp8BlockGemvComputationPrefix =
    "fp8_block_gemv_";

std::optional<Fp8BlockGemvSpec> MatchFp8BlockGemv(
    const HloFusionInstruction& fusion);

struct Fp8BlockGemvConfig {
  int64_t block_m;
  int64_t block_n;
  int64_t block_k;
  int num_warps;
  int num_stages;
};
std::optional<Fp8BlockGemvConfig> Fp8BlockGemvConfigFor(
    const HloScaledDotInstruction& dot,
    const se::GpuComputeCapability& gpu_version);

// Whether a backend here will claim this dot. Takes the compute capability
// because the answer depends on it: a batch that is neither one row nor a
// multiple of sixteen is servable only by the CUTLASS rung.
bool Fp8BlockGemvSupportsScaledDot(
    const HloScaledDotInstruction& dot,
    const se::GpuComputeCapability& gpu_version);

// A batch the Triton emitter cannot tile -- it reduces a single row or mma's a
// multiple of 16, and xtile.extract does not mask a partial tile. Such a batch
// is claimable only where HasCutlassBlockGemm holds.
bool Fp8BlockGemvBatchNeedsCutlass(int64_t batch);

// Whether this GPU has a vendored CUTLASS blockwise collective. Both
// Blackwell families do -- SM100 and SM120 take different ones -- and it is
// the only rung that tiles an arbitrary M.
bool HasCutlassBlockGemm(const se::GpuComputeCapability& gpu_version);

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> EmitFp8BlockGemvXTileModule(
    absl::string_view fn_name, const HloFusionInstruction& fusion,
    const Fp8BlockGemvSpec& spec,
    const xla::xtile::BlockLevelParameters& block_level_parameters,
    mlir::MLIRContext& mlir_context);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_TRITON_FP8_BLOCK_GEMV_H_
