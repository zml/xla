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

// Custom-call targets and predicates for the Apple Metal GPU backend's GEMM
// family -- the Metal peers of the cuBLAS/cuDNN ones in cublas_cudnn.h. Metal
// has no cuBLAS, so GemmRewriter emits these targets (for the first-class
// MetalComputeCapability) and ThunkEmitter routes them to the in-tree Metal
// thunks/kernels. Kept out of cublas_cudnn.h so the Metal backend doesn't live
// in a CUDA-named header.

// A general matrix multiplication run on Apple Metal via metalBLAS (no cuBLAS
// on Metal). GemmRewriter emits this for MetalComputeCapability; ThunkEmitter
// routes it to MetalGemmThunk.
inline constexpr absl::string_view kMetalGemmCallTarget = "__metal$gemm";

// Weight-only scaled matmul on Apple Metal: single custom-call target for every
// fused dequant×GEMM scheme the Metal arm of FusedScaledDotRewriter supports.
// Produced from a weight-only kScaledDot (itself from the xla.scaled_dot
// composite). Operands:
//   {x[M,K] bf16, w[N,K] quant, scale[...]} -> out[M,N] bf16
// ThunkEmitter dispatches by weight/scale dtypes and layout; the schemes are
// MetalScaledMatmulScheme below.
inline constexpr absl::string_view kMetalScaledMatmulCallTarget =
    "zml$scaled_matmul";

// The fused dequant×GEMM schemes the Metal scaled-matmul thunks implement.
//
// This enum is the contract between the rewriter that EMITS
// kMetalScaledMatmulCallTarget and the ThunkEmitter that LOWERS it, and both
// sides must classify through ClassifyMetalScaledMatmul rather than re-deriving
// the scheme from dtypes. A scheme the rewriter claims but no thunk implements
// is not a slow path, it is a compile failure: fusing deletes the kScaledDot,
// so ScaledDotRewriter's generic dequantize-and-Dot expansion is no longer
// there to fall back to.
//
// OCP microscaling (MX / e8m0 group-32) is deliberately absent: no model emits
// it -- zml/nn.zig is the only producer of the xla.scaled_dot composite and no
// checkpoint carries e8m0 scales -- so the arm and its thunk were removed.
// Because it is absent here, an e8m0 scaled dot is simply never fused and
// lowers through the generic expansion.
enum class MetalScaledMatmulScheme {
  // f4e2m1 w[N,K] + f8e4m3fn scale[N,K/16] -> MetalNvfp4MatmulThunk.
  kNvfp4Group16,
  // f8e4m3fn w[N,K] + bf16 scale[N/128,K/128] -> MetalFp8GemvThunk.
  kFp8Block128,
  // f8e4m3fn w[N,K] + bf16 scale[N,1] -> MetalFp8GemvThunk.
  kFp8PerChannel,
};

// Classifies a (weight, weight-scale) shape pair against the schemes above.
// nullopt means no Metal thunk implements this pair, so it must not be fused.
// Shapes only -- the caller checks dot dimension numbers (K minor) and any
// backend ABI limits separately.
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

// Matrix multiplication that runs via the Metal/metalBLAS GEMM thunk.
inline bool IsMetalGemm(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kCustomCall &&
         hlo.custom_call_target() == kMetalGemmCallTarget;
}

// A grouped block-scaled GEMV for mixture-of-experts on Apple Metal. Unlike the
// dense GEMMs above, these are *model-emitted only* -- MoE top-k routing has no
// dot for GemmRewriter to match, so the model selects experts and emits the call
// directly (like zml$gdn). Output row r is computed against expert weights
// w[expert_id[r]]. ThunkEmitter routes both flavors to MetalMoeGemvThunk.
//
// There is deliberately no fp8 flavor. A __metal$moe_gemm$f8 target and a
// 128x128 block-dequantizing kernel existed here, with no emitter in any repo in
// any commit; the MoE producer is zml/moe/metal.zig, whose QuantMode is
// enum { none, nvfp4 } and whose Backend.auto rejects f8e4m3fn for .metal
// outright. Adding fp8 MoE means adding the ZML emitter first.

// NVFP4 flavor of the grouped MoE GEMV: f4e2m1 weights + e4m3 group-16 scales.
//   {x_rows[R,K] bf16, w[E,N,K] f4e2m1, scale[E,N,K/16] f8e4m3fn,
//    expert_id[R] s32, w_global_scale[E] f32 (optional)} -> out[R,N] bf16
//
// The optional trailing operand is the compressed-tensors per-expert weight
// *encode divisor* g_ct. Both nvfp4 kernels fold 1/g_ct[expert_id[r]] into the
// weight's group scale, which is why it is passed rather than applied to x:
// pre-scaling x costs a gather plus a full read/write pass over x_rows per MoE
// call, and cannot fuse into this opaque custom call. Folding into the weight
// also keeps the f32 accumulator at output magnitude -- never divide a
// global-inflated output, which quantizes small components away. g_ct is not
// MLX's similarly named global_scale_w (an amax): mlx_global_scale_w =
// 2688 / g_ct. Do not both pass this operand and pre-divide x by g_ct.
// ThunkEmitter routes this call to MetalMoeGemvThunk (nvfp4_gather_qmv decode
// + nvfp4_gather_qmm_rhs prefill).
inline constexpr absl::string_view kMetalMoeGemmF4CallTarget =
    "__metal$moe_gemm$f4";

inline bool IsMetalMoeGemmF4(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kCustomCall &&
         hlo.custom_call_target() == kMetalMoeGemmF4CallTarget;
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

// Either of the two MoE GEMM custom-call targets (bf16 / nvfp4).
inline bool IsMetalMoeGemmAny(const HloInstruction& hlo) {
  return IsMetalMoeGemmF4(hlo) || IsMetalMoeGemmBf16(hlo);
}

// A native keyed stable sort on Apple Metal. Generic Sort has no lowerable
// Metal path -- the legacy LLVM bitonic emitter produces NVVM intrinsics that
// air-as cannot assemble -- so RewriteSortToMetalThunk rewrites a stablehlo
// Sort that sorts by ONE float key (bf16/f16/f32) along the minor-most axis,
// carrying iota index operands, into this call:
//   {values[rows, n]} -> (sorted_values[rows, n], sorted_indices[rows, n] s32)
// The permuted indices reproduce every iota result of the original Sort (all
// iota operands are arange along the sort axis). `opaque` is "desc"
// (descending) or "asc". ThunkEmitter routes it to MetalSortThunk (the vendored
// MLX merge sort). Stable: ties keep index-ascending order == XLA is_stable.
inline constexpr absl::string_view kMetalSortCallTarget = "metal$sort";

inline bool IsMetalSort(const HloInstruction& hlo) {
  return hlo.opcode() == HloOpcode::kCustomCall &&
         hlo.custom_call_target() == kMetalSortCallTarget;
}

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_METAL_CUSTOM_CALLS_H_
