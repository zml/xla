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

#ifndef XLA_SERVICE_GPU_METAL_CUSTOM_CALLS_H_
#define XLA_SERVICE_GPU_METAL_CUSTOM_CALLS_H_

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"

namespace xla {
namespace gpu {

inline constexpr absl::string_view kMetalGemmCallTarget = "__metal$gemm";

inline constexpr absl::string_view kMetalGemmF8CallTarget = "__metal$gemm$f8";

inline absl::string_view MetalBlockScaledGemmTarget(PrimitiveType weight_type) {
  switch (weight_type) {
    case F8E4M3FN:
      return kMetalGemmF8CallTarget;
    default:
      return absl::string_view();
  }
}

inline bool IsMetalGemm(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kCustomCall &&
         hlo.custom_call_target() == kMetalGemmCallTarget;
}

inline bool IsMetalFp8Gemm(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kCustomCall &&
         hlo.custom_call_target() == kMetalGemmF8CallTarget;
}

inline constexpr absl::string_view kMetalMoeGemmF8CallTarget =
    "__metal$moe_gemm$f8";

inline bool IsMetalMoeGemm(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kCustomCall &&
         hlo.custom_call_target() == kMetalMoeGemmF8CallTarget;
}

inline constexpr absl::string_view kMetalMoeGemmCallTarget = "__metal$moe_gemm";

inline bool IsMetalMoeGemmBf16(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kCustomCall &&
         hlo.custom_call_target() == kMetalMoeGemmCallTarget;
}

inline constexpr absl::string_view kMetalSortCallTarget = "metal$sort";

inline bool IsMetalSort(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kCustomCall &&
         hlo.custom_call_target() == kMetalSortCallTarget;
}

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_METAL_CUSTOM_CALLS_H_
