load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
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
