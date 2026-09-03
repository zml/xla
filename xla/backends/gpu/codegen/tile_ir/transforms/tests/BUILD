load("//xla:lit.bzl", "lit_test_suite")

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

lit_test_suite(
    name = "mlir_lit_tests",
    srcs = glob(["*.mlir"]),
    cfg = "//xla:lit.cfg.py",
    tools = [
        "//xla/backends/gpu/codegen/tile_ir/tools:xtile_to_tilebc",
        "@llvm-project//llvm:FileCheck",
    ],
)
