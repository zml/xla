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

#include <cstdint>
#include <optional>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

inline constexpr absl::string_view kMetalGemmCallTarget = "__metal$gemm";

inline constexpr absl::string_view kMetalScaledMatmulCallTarget =
    "zml$scaled_matmul";

enum class MetalScaledMatmulScheme {
  kNvfp4Group16,
  kFp8Block128,
  kFp8PerChannel,
};

inline std::optional<MetalScaledMatmulScheme> ClassifyMetalScaledMatmul(
    const Shape& weights, const Shape& scale) {
  if (weights.dimensions().size() != 2 || scale.dimensions().size() != 2) {
    return std::nullopt;
  }
  const int64_t n = weights.dimensions(0);
  const int64_t k = weights.dimensions(1);
  const int64_t scale_n = scale.dimensions(0);
  const int64_t scale_k = scale.dimensions(1);

  if (weights.element_type() == F4E2M1FN &&
      scale.element_type() == F8E4M3FN) {
    if (scale_n == n && scale_k != 0 && k == scale_k * 16) {
      return MetalScaledMatmulScheme::kNvfp4Group16;
    }
    return std::nullopt;
  }
  if (weights.element_type() == F8E4M3FN && scale.element_type() == BF16) {
    if (scale_n == n && scale_k == 1) {
      return MetalScaledMatmulScheme::kFp8PerChannel;
    }
    if (n % 128 == 0 && k % 128 == 0 && scale_n == n / 128 &&
        scale_k == k / 128) {
      return MetalScaledMatmulScheme::kFp8Block128;
    }
  }
  return std::nullopt;
}

inline bool IsMetalScaledMatmul(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kCustomCall &&
         hlo.custom_call_target() == kMetalScaledMatmulCallTarget;
}

inline bool IsMetalGemm(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kCustomCall &&
         hlo.custom_call_target() == kMetalGemmCallTarget;
}

inline constexpr absl::string_view kMetalMoeGemmF8CallTarget =
    "__metal$moe_gemm$f8";

inline bool IsMetalMoeGemm(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kCustomCall &&
         hlo.custom_call_target() == kMetalMoeGemmF8CallTarget;
}

inline constexpr absl::string_view kMetalMoeGemmF4CallTarget =
    "__metal$moe_gemm$f4";

inline bool IsMetalMoeGemmF4(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kCustomCall &&
         hlo.custom_call_target() == kMetalMoeGemmF4CallTarget;
}

inline constexpr absl::string_view kMetalMoeGemmCallTarget = "__metal$moe_gemm";

inline bool IsMetalMoeGemmBf16(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kCustomCall &&
         hlo.custom_call_target() == kMetalMoeGemmCallTarget;
}

inline bool IsMetalMoeGemmAny(const HloInstruction& hlo) {
  return IsMetalMoeGemm(hlo) || IsMetalMoeGemmF4(hlo) ||
         IsMetalMoeGemmBf16(hlo);
}

inline constexpr absl::string_view kMetalSortCallTarget = "metal$sort";

inline bool IsMetalSort(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kCustomCall &&
         hlo.custom_call_target() == kMetalSortCallTarget;
}

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_METAL_CUSTOM_CALLS_H_
