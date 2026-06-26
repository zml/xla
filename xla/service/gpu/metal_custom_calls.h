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

// Custom-call targets and predicates for the Apple Metal GPU backend's GEMM
// family -- the Metal peers of the cuBLAS/cuDNN ones in cublas_cudnn.h. Metal
// has no cuBLAS, so GemmRewriter emits these targets (for the first-class
// MetalComputeCapability) and ThunkEmitter routes them to the in-tree Metal
// thunks/kernels. Kept out of cublas_cudnn.h so the Metal backend doesn't live
// in a CUDA-named header.

// A general matrix multiplication run on Apple Metal via metalBLAS (no cuBLAS on
// Metal). GemmRewriter emits this for MetalComputeCapability; ThunkEmitter
// routes it to MetalGemmThunk.
inline constexpr absl::string_view kMetalGemmCallTarget = "__metal$gemm";

// A fused block-scaled FP8 GEMV on Apple Metal (no cuBLASLt FP8 on Metal), peer
// to CUDA's __cublas$lt$matmul$f8. GemmRewriter emits this for a decode-shaped
// dot over an f8e4m3fn weight + bf16 128-block scales; the call carries
// {x, w_f8, scale} and ThunkEmitter routes it to MetalFp8GemvThunk.
inline constexpr absl::string_view kMetalGemmF8CallTarget = "__metal$gemm$f8";

// Maps a block-scaled quantized weight element type to its Metal fused-GEMV
// custom-call target, or "" if the type has no Metal kernel. The single
// per-format extension point the GemmRewriter matcher keys off: add a case here
// (plus a kMetalGemm<F>CallTarget constant, an IsMetal<F>Gemm predicate, and a
// decode kernel/thunk) when adding a new quantized weight family such as MXFP4.
inline absl::string_view MetalBlockScaledGemmTarget(PrimitiveType weight_type) {
  switch (weight_type) {
    case F8E4M3FN:
      return kMetalGemmF8CallTarget;
    default:
      return absl::string_view();
  }
}

// Matrix multiplication that runs via the Metal/metalBLAS GEMM thunk.
inline bool IsMetalGemm(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kCustomCall &&
         hlo.custom_call_target() == kMetalGemmCallTarget;
}

// Fused block-scaled FP8 GEMV routed to MetalFp8GemvThunk (peer to CUDA's
// IsCublasLtMatmulF8). A distinct target from IsMetalGemm so the two never
// share an ABI.
inline bool IsMetalFp8Gemm(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kCustomCall &&
         hlo.custom_call_target() == kMetalGemmF8CallTarget;
}

// A grouped block-scaled FP8 GEMV for mixture-of-experts on Apple Metal. Unlike
// the dense GEMMs above, this one is *model-emitted only* -- MoE top-k routing
// has no dot for GemmRewriter to match, so the model selects experts and emits
// this call directly (like zml$gdn). The call carries
//   {x_rows[R,K] bf16, w[E,N,K] f8e4m3fn, scale[E,N/128,K/128] bf16,
//    expert_id[R] s32} -> out[R,N] bf16,
// where output row r is computed against expert weights w[expert_id[r]] (128x128
// block-dequantized). ThunkEmitter routes it to MetalMoeGemvThunk. The $f8
// suffix leaves room for a future $f4 MoE kernel.
inline constexpr absl::string_view kMetalMoeGemmF8CallTarget =
    "__metal$moe_gemm$f8";

inline bool IsMetalMoeGemm(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kCustomCall &&
         hlo.custom_call_target() == kMetalMoeGemmF8CallTarget;
}

// The bf16/f16 (un-quantized) sibling of the grouped MoE GEMV above: same
// grouped per-row-against-its-expert matmul and same MetalMoeGemvThunk, but the
// weights are bf16/f16 with no block scales. The call carries
//   {x_rows[R,K] bf16, w[E,N,K] bf16, expert_id[R] s32} -> out[R,N] bf16
// (no scale operand). Used by the LFM2 MoE path, which has no fp8 weights.
inline constexpr absl::string_view kMetalMoeGemmCallTarget = "__metal$moe_gemm";

inline bool IsMetalMoeGemmBf16(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kCustomCall &&
         hlo.custom_call_target() == kMetalMoeGemmCallTarget;
}

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_METAL_CUSTOM_CALLS_H_
