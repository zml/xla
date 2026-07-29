load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# The last backend-neutral step of the GPU tiled-codegen pipeline. It is its own
# package so that any backend consuming a tiled fusion can depend on it without
# depending on the Triton emitter, which is where this code used to live.
#
# Deliberately NOT compatible_with = get_compatible_with_portable(): the tiling
# helpers this needs (//xla/service/gpu/model:tiling_from_block_parameters) are
# not portable-compatible either.
cc_library(
    name = "xtile_module",
    srcs = ["xtile_module.cc"],
    hdrs = ["xtile_module.h"],
    deps = [
        "//xla:util",
        "//xla/backends/gpu/codegen/emitters/ir:xla_gpu",
        "//xla/codegen/emitters/ir:xla",
        "//xla/codegen/tiling/experimental:tile",
        "//xla/codegen/tiling/experimental:tiled_hlo",
        "//xla/codegen/tiling:symbolic_tile_analysis",
        "//xla/codegen/tiling:tiling_specification",
        "//xla/codegen/xtile/codegen:emitter_helpers",
        "//xla/codegen/xtile/codegen:fusion_emitter",
        "//xla/codegen/xtile/ir:xtile",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/service/gpu/model:block_level_parameters",
        "//xla/service/gpu/model:tiling_from_block_parameters",
        "//xla/service/gpu/model:triton_emitter_constraints",
        "//xla/service:decision",
        "//xla/service:instruction_fusion",
        "//xla/stream_executor:device_description",
        "//xla/tools:hlo_decomposer_lib",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMIRTransforms",
        "@llvm-project//mlir:TensorDialect",
        "@stablehlo//:stablehlo_ops",
    ],
)
