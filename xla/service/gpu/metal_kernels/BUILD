
load("//xla/util:build_defs.bzl", "embed_files", "embed_tree")

package(
    default_visibility = [
        "//xla/backends/gpu/runtime:__pkg__",
        "//xla/service/gpu:__subpackages__",
    ],
    licenses = ["notice"],
)

embed_files(
    name = "metalblas_shaders",
    srcs = ["@metalblas//:compiled_shaders"],
    cpp_namespace = "xla::gpu",
)

embed_tree(
    name = "mlx_include_tree",
    cpp_namespace = "xla::gpu",
    root_sentinel = "@mlx//:metal_kernels_root_sentinel",
    root_sentinel_path = "mlx/backend/metal/kernels/utils.h",
    tree = "@mlx//:metal_kernel_headers",
)

embed_files(
    name = "mlx_kernels",
    srcs = glob(["mlx_entries/*.h"]),
    cpp_namespace = "xla::gpu",
)

embed_files(
    name = "pagedattention_tiled",
    srcs = ["vendored/vllm/pagedattention_tiled.metal"],
    cpp_namespace = "xla::gpu",
)

embed_files(
    name = "flash_attn_kernels",
    srcs = [
        "custom/flash_attn_prefill.metal",
        "custom/flash_attn_vec.metal",
    ],
    cpp_namespace = "xla::gpu",
)

embed_files(
    name = "paged_attn_vec",
    srcs = ["custom/paged_attn_vec.metal"],
    cpp_namespace = "xla::gpu",
)

embed_files(
    name = "kv_write",
    srcs = ["custom/kv_write.metal"],
    cpp_namespace = "xla::gpu",
)

embed_files(
    name = "topk",
    srcs = ["custom/topk.metal"],
    cpp_namespace = "xla::gpu",
)

embed_files(
    name = "fp8_gemv",
    srcs = ["custom/fp8_gemv.metal"],
    cpp_namespace = "xla::gpu",
)

embed_files(
    name = "fp8_gemv_pc",
    srcs = ["custom/fp8_gemv_pc.metal"],
    cpp_namespace = "xla::gpu",
)

embed_files(
    name = "fp8_moe_gemv",
    srcs = ["custom/fp8_moe_gemv.metal"],
    cpp_namespace = "xla::gpu",
)

embed_files(
    name = "bf16_moe_gemv",
    srcs = ["custom/bf16_moe_gemv.metal"],
    cpp_namespace = "xla::gpu",
)

embed_files(
    name = "moe_argsort",
    srcs = ["custom/moe_argsort.metal"],
    cpp_namespace = "xla::gpu",
)

embed_files(
    name = "permute_rows",
    srcs = ["custom/permute_rows.metal"],
    cpp_namespace = "xla::gpu",
)

embed_files(
    name = "gdn_linear_attention",
    srcs = ["vendored/vllm/gdn/gdn_linear_attention.metal"],
    cpp_namespace = "xla::gpu",
)
