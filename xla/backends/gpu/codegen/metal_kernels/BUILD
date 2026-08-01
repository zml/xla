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












