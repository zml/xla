load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_binary(
    name = "cuda_tile_bytecode_smoke",
    srcs = ["cuda_tile_bytecode_smoke.cc"],
    deps = [
        "@cuda_tile//:bytecode_writer",
        "@cuda_tile//:dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
    ],
)

cc_library(
    name = "tileiras_compiler",
    srcs = ["tileiras_compiler.cc"],
    hdrs = ["tileiras_compiler.h"],
    deps = [
        "//xla:util",
        "//xla:xla_proto_cc",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:subprocess_compilation",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:subprocess",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_binary(
    name = "xtile_to_tilebc",
    srcs = ["xtile_to_tilebc.cc"],
    deps = [
        "//xla/backends/gpu/codegen/tile_ir/transforms:passes",
        "//xla/codegen/emitters/ir:xla",
        "//xla/codegen/emitters/transforms:passes",
        "//xla/codegen/xtile/ir:xtile",
        "@cuda_tile//:bytecode_writer",
        "@cuda_tile//:dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
        "@stablehlo//:stablehlo_ops",
    ],
)
