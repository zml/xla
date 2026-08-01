"""
Provides the list of patches applied to the pinned metalBLAS archive.

Each entry is one named fix against metalBLAS @ the commit in workspace.bzl.
Patches are applied by `tf_http_archive` via `ctx.patch(strip=1)`, which is
NON-FUZZY: a bump that disturbs a hunk's context ABORTS THE FETCH rather than
silently dropping our change. That loud failure is the entire point of pinning.
Keep the default -U3 context; do not shrink it to make a bump "easier".

When you add a patch, head it with the measurement that justifies it, the
upstream PR URL if one is filed, and the condition under which it can be
deleted.

===========================================================================
DO NOT IMPORT METALBLAS'S BUILD FLAGS. THE FLAG VECTOR IS OURS AND FROZEN.
===========================================================================

metalBLAS's shaders are compiled at RUNTIME by metalblas_gemm.cc with our own
flag vector (get_mb_epi() + one family header + -DMB_BUILD_<NAME>, -std=metal
of our choosing). Adopt upstream's SOURCE, never its build flags. Someone
"faithfully adopting upstream" who also adopts its Metal flags would silently
move the greedy golden across every dense Metal matmul at once, and -- because
these shaders are compiled at runtime, not by Bazel -- the bench is the only
thing in CI that would catch it. "Bit-identical to upstream binaries" is not an
acceptance criterion and never was. The only criterion is "bit-identical to OUR
current output" -- the greedy golden.

===========================================================================
BUMPING THE PIN
===========================================================================

The pin lives in workspace.bzl and is trustworthy for a concrete reason: four
of the five vendored shaders (mb_epi, gemv_bt, gemv_nt, gemv_t) are
byte-verbatim against b4dd324, and mpp_tensor.h reproduces as upstream + the
one MB_TOKCLAMP patch below. That byte-identity is what makes the archive a
faithful replacement for the previously-checked-in headers.

A bump is therefore a deliberate act with a bench attached, not a version
number edit. What it must clear:

  1. The fetch itself. The MB_TOKCLAMP patch applies NON-FUZZY (above), so a
     bump that disturbs mpp_tensor.h's context around any of the three anchors
     (the __TOKEN__ define block, the kernel-arg list, the tile-offset guard)
     aborts the fetch. That is the loud signal that upstream moved under the
     clamp -- re-derive it, do not shrink the context to force it through.
  2. The four verbatim shaders staying verbatim. If a bump makes any of
     mb_epi/gemv_bt/gemv_nt/gemv_t differ from the archive, the vendoring
     premise (bytes come from upstream, unedited) no longer holds for that file
     and it needs its own recorded patch -- do not hand-edit the fetched copy.
  3. Both goldens, bit-identically. gemma-4-26B-A4B-NVFP4 and Qwen3.6-27B-FP8.
     mpp_tensor is the primary dense-GEMM backend for bf16/f16/f32 matmuls, so a
     bump that moves either golden is a bug until proven otherwise; these
     shaders are compiled at runtime, so nothing else in CI will notice.
"""

metalblas_patch_list = [
    # MB_TOKCLAMP -- the prefill-row-clamp/stale-arena race fix carried on the
    # mpp_tensor.h dense-GEMM shader. On Metal there is no totally-ordered
    # stream, so the old host-side encode-time read of num_tokens raced the GPU
    # producer of that metadata; the fix clamps the M grid purely on-GPU (like
    # CUDA) by passing the real prompt length as a device pointer and
    # early-returning any M-tile entirely past it. This is three APPEND-ONLY
    # hunks at stable anchors (zero upstream lines changed), so it rebases onto a
    # future mpp_tensor.h trivially -- and if it does not, the non-fuzzy apply
    # aborts the fetch, which is the signal to re-derive it. Delete only if
    # upstream adopts an equivalent on-GPU clamp under the same __MB_TOKCLAMP__
    # token that metalblas_gemm.cc substitutes.
    "//third_party/metalblas:mb_tokclamp.patch",
]
