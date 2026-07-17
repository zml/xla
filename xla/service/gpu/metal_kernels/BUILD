# Metal GPU kernels for the XLA Metal backend.
#
# Every kernel is a real `.metal` / `.h` shader source embedded uniformly via
# `embed_files` (xxd -> a `get_<stem>()` accessor returning the source string),
# then assembled and compiled at runtime by the thunk via the Metal toolchain
# that uses it (see metal_air_toolchain.cc). Runtime `__TOKEN__` / `$N`
# substitution specializes a shader for the live shapes/dtypes before compile.
#
# Layout:
#   metalBLAS's GEMM/GEMV shaders are NOT checked in here: they come from the
#   pinned @metalblas archive (//third_party/metalblas:workspace.bzl) plus the
#   MB_TOKCLAMP patch, embedded by :metalblas_shaders below.
#   vendored/  - third-party-derived shaders. Each bundle documents whether it
#                is byte-verbatim or adapted; do not assume `vendored/` means a
#                file can be refreshed with an unreviewed copy:
#       vllm/       vllm-metal                      (verbatim kernels)
#   mlx_entries/ - our entry sources for the MLX-derived bundles. These #include
#                the pinned @mlx archive rather than copying it, so upstream's
#                bytes are never checked in here (see :mlx_include_tree) and
#                cannot be hand-edited. vendored/mlx/ USED to hold ~9,800 lines
#                of hand-flattened upstream copies with no recorded base; it is
#                gone, and this is what replaced it. Adding a copy of upstream
#                MLX text back to this package is a regression -- add an entry
#                source and, if you must change upstream's bytes, a patch in
#                //third_party/mlx:series.bzl.
#   custom/    - shaders written for this backend (the flash-attention family is
#                adapted from llama.cpp's kernel_flash_attn_ext_vec; kv_write and
#                topk are original).

load("//xla/util:build_defs.bzl", "embed_files", "embed_tree")

package(
    default_visibility = [
        "//xla/backends/gpu/runtime:__pkg__",
        "//xla/service/gpu:__subpackages__",
    ],
    licenses = ["notice"],
)

# --- metalBLAS GEMM/GEMV shaders (from the pinned @metalblas archive) --------
#
# The five shader headers are NOT checked in -- they come from the pinned
# @metalblas archive (//third_party/metalblas:workspace.bzl), which fetches
# github.com/Isalia20/metalBLAS at a recorded 40-hex and applies the append-only
# MB_TOKCLAMP patch to mpp_tensor.h (//third_party/metalblas:series.bzl). Four of
# the five (mb_epi, gemv_bt, gemv_nt, gemv_t) are byte-verbatim against the pin;
# mpp_tensor.h is upstream + that patch. The archive's :compiled_shaders
# filegroup lists exactly these five, so drift now fails loud at FETCH instead of
# rotting behind BUILD prose -- that loud-failure-on-drift is the whole reason
# for the pin.
#
# Each family header is `#ifdef MB_BUILD_<NAME>`-guarded and the only always-
# active header is the shared epilogue mb_epi.h. So metalblas_gemm.cc compiles a
# family by concatenating get_mb_epi() + the family header (verbatim) with the
# matching `-DMB_BUILD_<NAME>` — byte-identical to compiling the upstream binder
# with that one flag, but with no build-time include-inlining. embed_files names
# each accessor from the file's BASENAME (xla/util/build_defs.bzl), so the
# archive paths metalblas/shaders/<stem>.h yield the exact get_<stem>() the C++
# already calls: get_mb_epi(), get_mpp_tensor(), get_gemv_bt(), get_gemv_nt(),
# get_gemv_t() (metalblas_gemm.cc:70, 214, 390, 434, 439). The stems are flat and
# unique, so there is no basename collision (unlike @mlx, which needs a path-
# keyed include tree; metalBLAS shaders have zero cross-header includes and are
# embedded, not fed to the Metal compiler via -I).
#
# The archive's filegroup carries only the four MB_BUILD_* families plus the
# epilogue -- see //third_party/metalblas:metalblas.BUILD for why the other 11
# upstream shaders are excluded. Note MB_BUILD_SPLITK is NOT the live split-K:
# that is MLX's, under the different flag MB_BUILD_MLX_SPLITK
# (metalblas_gemm.cc:285), sourced from mlx_entries/.
embed_files(
    name = "metalblas_shaders",
    srcs = ["@metalblas//:compiled_shaders"],
    cpp_namespace = "xla::gpu",
)

# --- MLX's Metal kernel headers, verbatim from the pinned archive ------------
#
# Generates get_mlx_include_tree(): every mlx/backend/metal/kernels/**/*.h from
# the @mlx archive (//third_party/mlx:workspace.bzl), embedded PATH-KEYED and
# untouched. MetalIncludeRoot() writes it back out once per process and passes
# it to the Metal compiler as -I, so an entry source in mlx_entries/ below is
# compiled against upstream's own bytes.
#
# Path-keyed, not embed_files: MLX's includes are repo-root-relative, and its
# basenames collide across the layout (kernels/utils.h vs steel/utils.h,
# kernels/defines.h vs steel/defines.h, steel/gemm/params.h vs
# steel/conv/params.h, ...), which embed_files cannot express -- it names each
# accessor from the basename and throws the path away.
embed_tree(
    name = "mlx_include_tree",
    cpp_namespace = "xla::gpu",
    root_sentinel = "@mlx//:metal_kernels_root_sentinel",
    root_sentinel_path = "mlx/backend/metal/kernels/utils.h",
    tree = "@mlx//:metal_kernel_headers",
)

# --- MLX Steel / quantized shaders -------------------------------------------
#
# One get_<stem>() per bundle: get_mlx_steel_splitk() (the split-K GEMM family
# metalblas_gemm.cc concatenates after mb_epi), get_mlx_steel_qgemm() (Steel
# tiled block-FP8 / NVFP4 q-GEMM, dense + MoE gather entries),
# get_mlx_fp4_qmv(), get_mlx_sort().
#
# Every bundle is an entry source: a prologue of #includes into the pinned @mlx
# tree (resolved via -I, see :mlx_include_tree) plus the [[kernel]] entries and
# forks that are genuinely ours. Upstream's text is not copied here -- that is
# the invariant this directory exists to hold, and it is now whole: the
# hand-flattened vendored/mlx/ blobs are all retired.
#
# Stems must stay unique -- embed_files names its accessor from the basename, so
# two bundles claiming one stem is a duplicate-symbol build error rather than a
# silent shadow.
#
# Kept apart from :metalblas_shaders so "which vendored tree does this thunk pull
# in" is a Bazel label rather than a BUILD comment.
embed_files(
    name = "mlx_kernels",
    srcs = glob(["mlx_entries/*.h"]),
    cpp_namespace = "xla::gpu",
)

# --- vllm-metal paged-attention tiled kernel (vendored verbatim) ------------
# Generates get_pagedattention_tiled().
embed_files(
    name = "pagedattention_tiled",
    srcs = ["vendored/vllm/pagedattention_tiled.metal"],
    cpp_namespace = "xla::gpu",
)

# --- Custom flash-attention kernels (adapted from llama.cpp) ----------------
# Generates get_flash_attn_prefill() (fa_ext) and get_flash_attn_vec() (fa_vec +
# fa_vec_hc + fa_vec_reduce). Both cover any head_dim % 16 == 0 via __DK__/__DV__.
embed_files(
    name = "flash_attn_kernels",
    srcs = [
        "custom/flash_attn_prefill.metal",
        "custom/flash_attn_vec.metal",
    ],
    cpp_namespace = "xla::gpu",
)

# --- Custom paged-attention vector decode kernel (adapted from llama.cpp) ----
# Generates get_paged_attn_vec().
embed_files(
    name = "paged_attn_vec",
    srcs = ["custom/paged_attn_vec.metal"],
    cpp_namespace = "xla::gpu",
)

# --- Custom predicated KV-cache write kernel --------------------------------
# Generates get_kv_write().
embed_files(
    name = "kv_write",
    srcs = ["custom/kv_write.metal"],
    cpp_namespace = "xla::gpu",
)

# --- Custom radix-select TopK kernels ---------------------------------------
# Generates get_topk().
embed_files(
    name = "topk",
    srcs = ["custom/topk.metal"],
    cpp_namespace = "xla::gpu",
)

# --- Dense block-scaled FP8 decode GEMV (M==1) -------------------------------
# Generates get_fp8_gemv(). Reads f8e4m3fn weights + 128x128 bf16 block scales
# and dequantizes inline (one DRAM pass over the 1-byte weight).
embed_files(
    name = "fp8_gemv",
    srcs = ["custom/fp8_gemv.metal"],
    cpp_namespace = "xla::gpu",
)

# --- Dense per-channel FP8 decode GEMV (M==1) --------------------------------
# Generates get_fp8_gemv_pc(). Same as fp8_gemv but scale is one bf16 per
# output channel [N,1] (factors out of the K reduction).
embed_files(
    name = "fp8_gemv_pc",
    srcs = ["custom/fp8_gemv_pc.metal"],
    cpp_namespace = "xla::gpu",
)

# NOTE: the MLX Steel tiled block-FP8 GEMM lives in mlx_entries/mlx_steel_qgemm.h
# (dense fp8_qmm_t entries; MLX's simdgroup_matrix Steel q-GEMM with the dequant
# swapped to our 128x128-block f8e4m3fn scheme). The MoE gather entries in that
# same bundle are bf16 and nvfp4 only -- fp8 MoE has no ZML emitter. It is
# embedded by the :mlx_kernels glob above (mlx_entries/*.h) as
# get_mlx_steel_qgemm() -- do NOT add a second embed target for it (duplicate
# symbol). Thunks depend on :mlx_kernels and call get_mlx_steel_qgemm().

# --- Custom grouped bf16 GEMV for MoE experts --------------------------------
# Generates get_bf16_moe_gemv(). Each output row indexes its own expert weight
# matrix (expert_id[r]) -- the per-expert GEMM for top-k mixture-of-experts
# decode/prefill. bf16 weights, no block scales (e.g. LFM2). The NVFP4 flavor is
# MLX's nvfp4_gather_qmv in :mlx_kernels; there is no fp8 flavor.
embed_files(
    name = "bf16_moe_gemv",
    srcs = ["custom/bf16_moe_gemv.metal"],
    cpp_namespace = "xla::gpu",
)

# --- Custom counting-sort (group routed rows by expert) for sorted MoE prefill -
# Generates get_moe_argsort(). Single-threadgroup counting sort that groups the
# R routed rows by expert id so the MLX gather q-GEMM's per-tile run-walk reuses
# each expert's weight across its (now contiguous) rows -- the large-R (prefill)
# MoE GEMM path, where each expert serves many rows.
embed_files(
    name = "moe_argsort",
    srcs = ["custom/moe_argsort.metal"],
    cpp_namespace = "xla::gpu",
)

# --- Sorted MoE route permutation --------------------------------------------
# Generates get_permute_rows(): gather_rows and scatter_rows, which permute the
# routed rows into expert-contiguous order for the sorted-prefill MoE path and
# permute the result back (MetalMoeGemvThunk).
embed_files(
    name = "permute_rows",
    srcs = ["custom/permute_rows.metal"],
    cpp_namespace = "xla::gpu",
)

# --- vllm-metal Gated DeltaNet linear-attention kernel (vendored verbatim) ---
# Generates get_gdn_linear_attention(). The recurrent delta-rule kernel for
# Qwen3-Next style hybrid models (the Triton GDN recurrence has no Metal path).
embed_files(
    name = "gdn_linear_attention",
    srcs = ["vendored/vllm/gdn/gdn_linear_attention.metal"],
    cpp_namespace = "xla::gpu",
)
