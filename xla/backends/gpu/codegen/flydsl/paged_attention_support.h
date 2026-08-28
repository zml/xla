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

#ifndef XLA_BACKENDS_GPU_CODEGEN_FLYDSL_PAGED_ATTENTION_SUPPORT_H_
#define XLA_BACKENDS_GPU_CODEGEN_FLYDSL_PAGED_ATTENTION_SUPPORT_H_

#include <cstdint>
#include <optional>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu::flydsl {

inline constexpr char kFlyPagedAttentionDecodeCallTarget[] =
    "__fly$paged_attention_decode";
inline constexpr char kFlyPagedAttentionSegmentedProducerCallTarget[] =
    "__fly$paged_attention_decode_segmented_producer";
inline constexpr char kFlyPagedAttentionSegmentedReducerCallTarget[] =
    "__fly$paged_attention_decode_segmented_reducer";
inline constexpr char kFlyPagedAttentionSegmentedFusedCallTarget[] =
    "__fly$paged_attention_decode_segmented_fused";

// Static properties of the first native unified-attention specialization.
// The public custom-call operands are Q, K cache, V cache, used-K lengths,
// block table, and a scalar compile-time softmax scale. The rewriter keeps the
// scale in the fused computation, so the emitted kernel has five inputs.
struct FlyPagedAttentionDescriptor {
  int64_t sequences;
  int64_t query_heads;
  int64_t kv_heads;
  int64_t gqa_group;
  int64_t head_dimension;
  int64_t cache_blocks;
  int64_t page_size;
  int64_t pages_per_sequence;
  int64_t max_context;
  double scale;
  PrimitiveType element_type;
  const HloInstruction* query;
  const HloInstruction* key_cache;
  const HloInstruction* value_cache;
  const HloInstruction* used_k;
  const HloInstruction* block_table;
  const HloInstruction* call;
};

// The segmented producer preserves the public paged-attention operands and
// matches FlyDSL's native reduction ABI: unnormalized FP32 output, log2-domain
// maximum, and exponential sum for every independently scheduled context
// segment. Keep those in three separate buffers: [B,H,S,D], [B,H,S], and
// [B,H,S].
struct FlyPagedAttentionSegmentedProducerDescriptor {
  FlyPagedAttentionDescriptor attention;
  int64_t num_segments;
  int64_t segment_tokens;
  // The fused form returns final output, partial output/max/sum, and a compact
  // two-word U32 generation/count completion-ticket buffer. Its last producer
  // workgroup performs the online reduction, avoiding a second kernel launch
  // and a separate ticket-clear dispatch.
  bool fused_reducer;
};

struct FlyPagedAttentionSegmentedReducerDescriptor {
  int64_t sequences;
  int64_t query_heads;
  int64_t num_segments;
  int64_t head_dimension;
  PrimitiveType element_type;
  const HloInstruction* partial_output;
  const HloInstruction* partial_maximum;
  const HloInstruction* partial_sum;
  const HloInstruction* call;
};

// Recognizes the F16/BF16, GQA1-16, D128 paged-decode contract used by the
// native gfx942 two-phase and streaming-online kernels.
std::optional<FlyPagedAttentionDescriptor> GetFlyPagedAttentionDescriptor(
    const HloInstruction& call);

std::optional<FlyPagedAttentionDescriptor> GetFlyPagedAttentionDescriptor(
    const HloFusionAnalysis& analysis);

std::optional<FlyPagedAttentionSegmentedProducerDescriptor>
GetFlyPagedAttentionSegmentedProducerDescriptor(const HloInstruction& call);

std::optional<FlyPagedAttentionSegmentedProducerDescriptor>
GetFlyPagedAttentionSegmentedProducerDescriptor(
    const HloFusionAnalysis& analysis);

std::optional<FlyPagedAttentionSegmentedReducerDescriptor>
GetFlyPagedAttentionSegmentedReducerDescriptor(const HloInstruction& call);

std::optional<FlyPagedAttentionSegmentedReducerDescriptor>
GetFlyPagedAttentionSegmentedReducerDescriptor(
    const HloFusionAnalysis& analysis);

}  // namespace xla::gpu::flydsl

#endif  // XLA_BACKENDS_GPU_CODEGEN_FLYDSL_PAGED_ATTENTION_SUPPORT_H_
