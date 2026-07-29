load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

gentbl_cc_library(
    name = "passes_inc_gen",
    tbl_outs = {
        "passes.h.inc": [
            "-gen-pass-decls",
            "-name=TileIrTransforms",
        ],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    visibility = ["//visibility:private"],
    deps = ["@llvm-project//mlir:PassBaseTdFiles"],
)

cc_library(
    name = "passes",
    srcs = ["xtile_lower_to_cuda_tile.cc"],
    hdrs = ["passes.h"],
    deps = [
        ":passes_inc_gen",
        "//xla/codegen/emitters/ir:xla",
        "//xla/codegen/xtile/codegen:emitter_helpers",
        "//xla/codegen/xtile/ir:xtile",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@cuda_tile//:CudaTileDialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
    ],
)
