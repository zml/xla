load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "tileiras_compiler",
    srcs = ["tileiras_compiler.cc"],
    hdrs = ["tileiras_compiler.h"],
    tags = ["cuda-only"],
    deps = [
        "//xla:util",
        "//xla:xla_proto_cc",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:semantic_version",
        "//xla/stream_executor/cuda:subprocess_compilation",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:subprocess",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
    ],
)

# The only target that links @cuda_tile: textual cuda_tile MLIR in, Tile IR
# bytecode out. Pure MLIR, no CUDA toolkit dependency.
cc_library(
    name = "tile_ir_module",
    srcs = ["tile_ir_module.cc"],
    hdrs = ["tile_ir_module.h"],
    deps = [
        "//xla/mlir/utils:error_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@cuda_tile//:CudaTileBytecode",
        "@cuda_tile//:CudaTileDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
    ],
)

xla_cc_test(
    name = "tile_ir_module_test",
    srcs = ["tile_ir_module_test.cc"],
    deps = [
        ":tile_ir_module",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@cuda_tile//:CudaTileBytecode",
        "@cuda_tile//:CudaTileDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "fusion",
    srcs = ["fusion.cc"],
    hdrs = ["fusion.h"],
    visibility = [
        "//xla/backends/gpu/codegen:__subpackages__",
        "//xla/service/gpu:__subpackages__",
    ],
    deps = [
        ":tileiras_compiler",
        "//xla/backends/gpu/codegen:fusion_emitter",
        "//xla/backends/gpu/codegen/kernels:custom_kernel",
        "//xla/backends/gpu/codegen/kernels:ptx_custom_kernel",
        "//xla/backends/gpu/codegen/tile_ir/transforms:passes",
        "//xla/backends/gpu/codegen/triton:fp8_block_gemv",
        "//xla/backends/gpu/codegen/xtile:xtile_module",
        "//xla/backends/gpu/runtime:custom_kernel_thunk",
        "//xla/backends/gpu/runtime:thunk",
        "//xla/codegen/emitters:computation_fingerprint",
        "//xla/codegen/emitters:kernel_arguments",
        "//xla/codegen/emitters/transforms:passes",
        "//xla/codegen/xtile:block_level_parameters",
        "//xla/codegen/xtile/ir:xtile",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_query",
        "//xla/service:dump",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:hlo_fusion_analysis",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/service/gpu:ir_emitter_context",
        "//xla/service/gpu:kernel_reuse_cache",
        "//xla/service/gpu:launch_dimensions",
        "//xla/stream_executor:launch_dim",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@cuda_tile//:CudaTileBytecode",
        "@cuda_tile//:CudaTileDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)
