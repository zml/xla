# Metal GPU kernels for the XLA Metal backend.
#
# Every kernel is a real `.metal` / `.h` shader source embedded uniformly via
# `embed_files` (xxd -> a `get_<stem>()` accessor returning the source string),
# then assembled and compiled at runtime by the thunk via the Metal toolchain
# that uses it (see metal_air_toolchain.cc). Runtime `__TOKEN__` / `$N`
# substitution specializes a shader for the live shapes/dtypes before compile.
#
# Layout:
#   vendored/  - third-party shaders, copied VERBATIM from upstream (update = cp):
#       metalblas/  github.com/Isalia20/metalBLAS  (the GEMM/GEMV family + binder)
#       mlx/        github.com/ml-explore/mlx       (steel split-K GEMM, MIT)
#       vllm/       vllm-metal                      (paged-attention tiled kernel)
#   custom/    - shaders written for this backend (the flash-attention family is
#                adapted from llama.cpp's kernel_flash_attn_ext_vec; kv_write and
#                topk are original).

load("//xla/util:build_defs.bzl", "embed_files")

package(
    default_visibility = [
        "//xla/backends/gpu/runtime:__pkg__",
        "//xla/service/gpu:__subpackages__",
    ],
    licenses = ["notice"],
)

# --- metalBLAS GEMM/GEMV shaders (vendored verbatim) ------------------------
#
# Each family header is `#ifdef MB_BUILD_<NAME>`-guarded and the only always-
# active header is the shared epilogue mb_epi.h. So metalblas_gemm.cc compiles a
# family by concatenating get_mb_epi() + the family header (verbatim) with the
# matching `-DMB_BUILD_<NAME>` — byte-identical to compiling the upstream binder
# with that one flag, but with no build-time include-inlining. Embed every shader
# header individually (one get_<stem>() per file: get_mb_epi(), get_mpp_tensor(),
# get_gemv_bt(), get_mlx_steel_splitk(), ...). Updating the bundle stays a plain
# `cp` of upstream shaders/.
embed_files(
    name = "metalblas_shaders",
    srcs = glob([
        "vendored/metalblas/*.h",
        "vendored/mlx/*.h",
    ]),
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
# Generates get_flash_attn_serial(), get_flash_attn_vec(), get_flash_attn_prefill().
embed_files(
    name = "flash_attn_kernels",
    srcs = [
        "custom/flash_attn_prefill.metal",
        "custom/flash_attn_serial.metal",
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

# --- Custom fused FP8 (block-scaled) GEMV decode kernel ----------------------
# Generates get_fp8_gemv(). Reads f8e4m3fn weights + 128x128 bf16 block scales
# inline so decode matmuls touch DRAM once over the 1-byte weight, instead of
# the generic reduce-fusion that the f8->bf16 dequant prologue otherwise forces.
embed_files(
    name = "fp8_gemv",
    srcs = ["custom/fp8_gemv.metal"],
    cpp_namespace = "xla::gpu",
)

# --- Tiled block-scaled FP8 thin GEMM for batched decode (M = batch > 1) ------
# Generates get_fp8_gemm_tiled(). Same block-scaled f8 decode as fp8_gemv, but
# reuses each weight byte across all M rows (threadgroup weight tile) instead of
# re-reading the weight once per row -- removes the M-x DRAM-traffic blowup that
# makes the per-row fp8_gemv collapse at batch>1.
embed_files(
    name = "fp8_gemm_tiled",
    srcs = ["custom/fp8_gemm_tiled.metal"],
    cpp_namespace = "xla::gpu",
)

# NOTE: the MLX Steel tiled block-FP8 GEMM lives in vendored/mlx/mlx_steel_qgemm.h
# (entries fp8_qmm_t dense + fp8_gather_qmm_rhs MoE; MLX's simdgroup_matrix Steel
# q-GEMM with the dequant swapped to our 128x128-block f8e4m3fn scheme). It is
# embedded by the metalblas_shaders glob above (vendored/mlx/*.h) as
# get_mlx_steel_qgemm() -- do NOT add a second embed target for it (duplicate
# symbol). Thunks depend on :metalblas_shaders and call get_mlx_steel_qgemm().

# --- Custom grouped FP8 (block-scaled) GEMV for MoE experts ------------------
# Generates get_fp8_moe_gemv(). Same block-scaled f8 decode as fp8_gemv, but
# each output row indexes its own expert weight matrix (expert_id[r]) -- the
# per-expert GEMM for top-k mixture-of-experts decode/prefill.
embed_files(
    name = "fp8_moe_gemv",
    srcs = ["custom/fp8_moe_gemv.metal"],
    cpp_namespace = "xla::gpu",
)

# --- bf16 (un-quantized) twin of fp8_moe_gemv for fp8-free MoE (e.g. LFM2) -----
# Generates get_bf16_moe_gemv(). Same grouped per-expert GEMV, bf16 weights, no
# block scales.
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

# --- Custom row gather/scatter for the sorted MoE prefill path ----------------
# Generates get_permute_rows() (gather_rows + scatter_rows): permute routed rows
# into expert-sorted order before the gather q-GEMM, scatter the result back.
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
