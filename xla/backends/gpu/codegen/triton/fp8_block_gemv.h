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

namespace xla::gpu {

struct Fp8BlockGemvSpec {
  int64_t activation_index;
  int64_t weight_index;
  int64_t scale_index;
  int64_t n;
  int64_t k;
  int64_t batch;
  bool activation_batch_major;
};

inline constexpr int64_t kMaxFp8BlockGemvBatch = 16;

std::optional<Fp8BlockGemvSpec> MatchFp8BlockGemv(
    const HloFusionInstruction& fusion);

struct Fp8BlockGemvConfig {
  int64_t block_n;
  int64_t block_k;
  int num_warps;
  int num_stages;
};
std::optional<Fp8BlockGemvConfig> Fp8BlockGemvConfigFor(
    const HloScaledDotInstruction& dot);

bool Fp8BlockGemvSupportsScaledDot(const HloScaledDotInstruction& dot);

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> EmitFp8BlockGemvXTileModule(
    absl::string_view fn_name, const HloFusionInstruction& fusion,
    const Fp8BlockGemvSpec& spec,
    const xla::xtile::BlockLevelParameters& block_level_parameters,
    mlir::MLIRContext& mlir_context);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_TRITON_FP8_BLOCK_GEMV_H_
