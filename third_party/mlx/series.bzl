"""
Provides the list of patches applied to the pinned MLX archive.

Each entry is one named fix against MLX @ the commit in workspace.bzl. Patches
are applied by `tf_http_archive` via `ctx.patch(strip=1)`, which is NON-FUZZY:
a bump that disturbs a hunk's context ABORTS THE FETCH rather than silently
dropping our change. That loud failure is the entire point of pinning. Keep the
default -U3 context; do not shrink it to make a bump "easier".

When you add a patch, head it with the measurement that justifies it, the
upstream PR URL if one is filed, and the condition under which it can be
deleted. `git apply --stat` over this list is the fork table -- derive it, never
write it down, or it will rot like everything else here has.

===========================================================================
DO NOT IMPORT MLX'S BUILD FLAGS. THE FLAG VECTOR IS OURS AND IT IS FROZEN.
===========================================================================

MLX AOT-compiles its Metal kernels with `-fno-fast-math`
(mlx/backend/metal/kernels/CMakeLists.txt). We pass only `-std=metal4.0`, i.e.
fast-math ON, and we always have. Measured: the same source with and without
`-fno-fast-math` emits different AIR (8000 vs 7888 bytes, differing from byte
13).

So we were NEVER bit-identical to upstream MLX's binaries, and that is fine.
"Bit-identical to upstream" is not an acceptance criterion and never was. The
only criterion is "bit-identical to OUR current output" -- the greedy golden.
Someone "faithfully adopting upstream" who also adopts its CMake flags will
silently move that golden across every Metal model at once, and the bench is
the only thing that would catch it. Adopt upstream's SOURCE. Never its flags.

===========================================================================
BUMPING THE PIN
===========================================================================

The pin lives in workspace.bzl and was RECOVERED, not chosen: MLX v0.32.1 is
the commit that regenerates the previously-checked-in hand-flattened blobs
byte-identically, which is how the base of a fork nobody had recorded was
established in the first place. That is the only reason it is trustworthy.

A bump is therefore a deliberate act with a bench attached, not a version
number edit. What it must clear:

  1. The fetch itself. Patches apply NON-FUZZY (above), so a disturbed hunk
     aborts the fetch. With an empty series there is nothing to abort, which
     means a bump currently fails LOUDLY nowhere -- see (2).
  2. The rename-forks. mlx_entries/ carries xla_-prefixed forks of upstream
     bodies (xla_fp_qmv_impl, xla_fp_qmv_fast_impl, xla_fp_qmv_wide_impl,
     XlaQuantizedBlockLoader). Upstream's originals are compiled but never
     called. A rename cannot collide on a bump and cannot silently un-apply --
     that is exactly why they are renames -- but it also means a bump will NOT
     tell you upstream changed underneath them. Re-read each XLA DELTA note and
     decide whether the fork is still needed or now collapses into a deletion.
  3. Both goldens, bit-identically. gemma-4-26B-A4B-NVFP4 and Qwen3.6-27B-FP8.
     A bump that moves either is a bug until proven otherwise; these bundles are
     compiled at runtime, so nothing else in CI will notice.
"""

mlx_patch_list = [
    # EMPTY, AND THAT IS A RESULT, NOT AN OMISSION. Every bundle is compiled
    # against the archive and the series stayed at zero. The plan expected two
    # patches on day one (k_aligned, the inv_g/ScaleDecoder seam); both were
    # dropped once the fp4 bundle was actually built, for one measured reason.
    #
    # A patch to upstream's SOURCE only does something if we COMPILE upstream's
    # source. For the qmv family we do not. Our fp_qmv_impl / fp_qmv_fast_impl /
    # fp_qmv_wide_impl diverge from upstream's in the BODY, in ways the plan
    # itself concluded must stay forked (the per-tile partial guard, the int64_t
    # address widening) -- so they are rename-forks (xla_*) in
    # metal_kernels/mlx_entries/mlx_fp4_qmv.h, and upstream's originals are
    # compiled but never called. Patching a body nothing calls buys nothing and
    # costs a hunk that can rot. Same for the Steel q-GEMM bundle's
    # XlaQuantizedBlockLoader, and same for the ScaleDecoder seam, which has no
    # upstream body to thread into while the forks stand.
    #
    # k_aligned deserves care: it is a REAL upstream defect (the safe K-tail
    # bodies index thread U x_thread[] with a runtime bound, defeating SROA for
    # the whole function -- worth 90.8 -> 97.3 tok/s), and it is live in
    # upstream's fp_qmv_impl. It is still not a patch, and the reason is now
    # simply that nothing we compile calls that body: our fork omits the tail
    # outright (the emitter rejects K%16!=0, so it cannot occur), and upstream's
    # fp_qmv_impl comes in via the include unused. This note used to say the
    # patch would move "the MXFP path, which is de-scoped and ungateable" -- that
    # path is gone entirely now (no model emits e8m0; the mxfp bundle, its thunk
    # and its entries were deleted), so a patch here would move nothing at all.
    #
    # So: FILE THESE UPSTREAM, do not carry them here. The PRs worth writing are
    # k_aligned, QuantizedBlockLoader::load_safe's per-pack K-tail guard (a
    # genuine correctness bug for any rectangular tile), the int64_t widening,
    # and the defaulted ScaleDecoder seam. If any lands, the matching rename-fork
    # in mlx_entries/ collapses into a deletion -- which is the whole point of
    # having a recorded base. TODOs sit at each fork site.
    #
    # When the first patch does arrive, head it with the measurement that
    # justifies it and the condition under which it can be deleted, per the note
    # above.
]
