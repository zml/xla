load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla/util:build_defs.bzl", "embed_files", "embed_tree")

package(
    default_visibility = [
        "//xla/backends/gpu:__subpackages__",
        "//xla/service/gpu:__subpackages__",
        "//xla/stream_executor/metal:__pkg__",
    ],
    licenses = ["notice"],
)

# Materializes the embedded MLX include tree on disk and hands the Metal
# compiler its -I root.
cc_library(
    name = "metal_include_root",
    srcs = ["metal_include_root.cc"],
    hdrs = ["metal_include_root.h"],
    tags = [
        "gpu",
        "metal-only",
    ],
    deps = [
        ":mlx_include_tree",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/util:embedded_file",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:fingerprint",
    ],
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
