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

#ifndef XLA_BACKENDS_GPU_AUTOTUNER_TRITON_AITER_UNIFIED_ATTENTION_H_
#define XLA_BACKENDS_GPU_AUTOTUNER_TRITON_AITER_UNIFIED_ATTENTION_H_

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/codegen/flydsl/paged_attention_support.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/backend_configs.pb.h"

namespace xla::gpu {

// Returns the segmented paged-attention descriptor when `instruction` is the
// Fly producer fusion that the embedded AITER Triton kernel can replace.
absl::StatusOr<flydsl::FlyPagedAttentionSegmentedProducerDescriptor>
GetAiterTritonPagedAttentionDescriptor(const HloInstruction& instruction);

// Materializes AITER's unified-attention TTIR for the concrete XLA buffer
// shapes. The producer epilogue is adapted to Fly's normalized partial-state
// ABI so both producers can safely share and autotune against one reducer.
absl::StatusOr<std::string> BuildAiterTritonPagedAttentionTtir(
    const flydsl::FlyPagedAttentionSegmentedProducerDescriptor& descriptor);

// Replaces a Fly segmented-producer fusion with an embedded Triton custom call
// configured with AITER's unified-attention schedule.
absl::Status ApplyAiterTritonPagedAttentionConfig(
    HloInstruction& instruction, const BlockLevelFusionConfig& config);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_AUTOTUNER_TRITON_AITER_UNIFIED_ATTENTION_H_
