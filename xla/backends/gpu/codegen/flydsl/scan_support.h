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

#ifndef XLA_BACKENDS_GPU_CODEGEN_FLYDSL_SCAN_SUPPORT_H_
#define XLA_BACKENDS_GPU_CODEGEN_FLYDSL_SCAN_SUPPORT_H_

#include <cstdint>
#include <optional>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu::flydsl {

inline constexpr char kFlyScanCallTarget[] = "__fly$scan";

// Static properties of a physically-minor inclusive additive scan. Rows are
// contiguous in physical memory regardless of the logical dimension order.
struct FlyScanDescriptor {
  int64_t row_length;
  int64_t rows;
  bool is_reverse;
  PrimitiveType element_type;
  const HloInstruction* input;
  const HloInstruction* call;
};

// The first native implementation assigns one Wave64 to a row and unrolls a
// bounded number of striped prefix tiles. Larger rows remain on rocPRIM until
// the hierarchical multi-kernel implementation is available.
bool IsFlyScanSupported(const Shape& shape, const CubScanOptions& options);

std::optional<FlyScanDescriptor> GetFlyScanDescriptor(
    const HloInstruction& call);

std::optional<FlyScanDescriptor> GetFlyScanDescriptor(
    const HloFusionAnalysis& analysis);

}  // namespace xla::gpu::flydsl

#endif  // XLA_BACKENDS_GPU_CODEGEN_FLYDSL_SCAN_SUPPORT_H_
