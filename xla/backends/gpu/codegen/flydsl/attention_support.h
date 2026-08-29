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

#ifndef XLA_BACKENDS_GPU_CODEGEN_FLYDSL_ATTENTION_SUPPORT_H_
#define XLA_BACKENDS_GPU_CODEGEN_FLYDSL_ATTENTION_SUPPORT_H_

#include <array>
#include <cstdint>
#include <optional>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu::flydsl {

// Static properties of the dense packed-QKV attention region formed by
// AttentionRewriterFly. The rank-2 parameter stores Q, K, and V consecutively
// for each token. MHA uses H query and H key/value heads; GQA uses H query and
// a smaller number of key/value heads. Output is logically [B,S,H,D].
struct FlyAttentionDescriptor {
  int64_t batch;
  // `sequence` is the query length. Packed self-attention has the same key/
  // value length; cross-attention may use a different one and two parameters.
  int64_t sequence;
  int64_t key_value_sequence;
  int64_t heads;
  int64_t key_value_heads;
  int64_t head_dimension;
  double scale;
  bool causal;
  PrimitiveType element_type;
  const HloInstruction* qkv_parameter;
  const HloInstruction* key_value_parameter;
  // Optional additive score bias, addressed logically as [B,H,Q,K]. A zero
  // stride denotes a broadcast dimension.
  const HloInstruction* bias_parameter;
  std::array<int64_t, 4> bias_strides;
  // Optional predicate score mask, also addressed as [B,H,Q,K]. False values
  // are replaced by negative infinity before online softmax.
  const HloInstruction* mask_parameter;
  std::array<int64_t, 4> mask_strides;
  const HloInstruction* qk_dot;
  const HloInstruction* pv_dot;
  const HloInstruction* softmax_root;
};

// Returns the unmasked scores when `input` is the canonical lower-triangular
// causal mask emitted by XLA: broadcast(iota(key) <= iota(query)) followed by
// select(predicate, scores, -inf). Returns nullptr for any other expression.
const HloInstruction* GetFlyCausalMaskScores(const HloInstruction& input,
                                             int64_t sequence);

// As above, for layouts where the logical query/key dimensions are not the
// final two physical dimensions (for example grouped-query attention).
const HloInstruction* GetFlyCausalMaskScoresAlongDimensions(
    const HloInstruction& input, int64_t sequence, int64_t query_dimension,
    int64_t key_dimension);

// Recognizes a dense F16/BF16 attention region with canonical stable softmax,
// optional causal masking, and packed QKV input. The matcher validates
// semantic graph structure rather than relying on instruction names.
std::optional<FlyAttentionDescriptor> GetFlyAttentionDescriptor(
    const HloFusionAnalysis& analysis);

}  // namespace xla::gpu::flydsl

#endif  // XLA_BACKENDS_GPU_CODEGEN_FLYDSL_ATTENTION_SUPPORT_H_
