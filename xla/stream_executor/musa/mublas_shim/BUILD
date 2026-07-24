"""Versioned, vendor-linked adapter for the MUSA muBLAS library."""

load(
    "@local_config_musa//musa:build_defs.bzl",
    "if_musa",
)
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "mublas_shim_abi",
    hdrs = ["mublas_shim_abi.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mublas_shim_float16",
    hdrs = ["mublas_shim_float16.h"],
)

# The PJRT plugin discovers this DSO next to itself at runtime. The vendor SDK
# is intentionally confined to this target and must never become a plugin link
# dependency.
cc_binary(
    name = "libxla_musa_mublas_shim.so",
    srcs = ["mublas_shim.cc"],
    additional_linker_inputs = ["mublas_shim_version_script.lds"],
    copts = ["-fvisibility=hidden"],
    linkopts = [
        "-Wl,--no-undefined",
        "-Wl,--version-script,$(location mublas_shim_version_script.lds)",
        "-Wl,-rpath,$$ORIGIN",
        "-Wl,-soname,libxla_musa_mublas_shim.so.1",
    ],
    linkshared = True,
    tags = [
        "gpu",
        "musa-only",
    ],
    target_compatible_with = if_musa([], ["@platforms//:incompatible"]),
    visibility = ["//visibility:public"],
    deps = [
        ":mublas_shim_abi",
        ":mublas_shim_float16",
        "@local_config_musa//musa:mublas",
    ],
)

xla_cc_test(
    name = "mublas_shim_abi_test",
    srcs = ["mublas_shim_abi_test.cc"],
    deps = [
        ":mublas_shim_abi",
        ":mublas_shim_float16",
        "@com_google_googletest//:gtest_main",
    ],
)
