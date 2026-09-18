"""Versioned, vendor-linked adapter for the MUSA muDNN library."""

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
    name = "mudnn_shim_abi",
    hdrs = ["mudnn_shim_abi.h"],
    visibility = ["//visibility:public"],
)

# Embedded as data in PJRT. Vendor headers and link edges remain confined to
# this DSO; the core plugin does not link the vendor C++ ABI.
cc_binary(
    name = "libxla_musa_mudnn_shim.so",
    srcs = ["mudnn_shim.cc"],
    additional_linker_inputs = ["mudnn_shim_version_script.lds"],
    copts = ["-fvisibility=hidden"],
    linkopts = [
        "-Wl,--disable-new-dtags",
        "-Wl,--no-undefined",
        "-Wl,--version-script,$(location mudnn_shim_version_script.lds)",
        "-Wl,-rpath,$$ORIGIN/musa-sdk/lib:$$ORIGIN",
        "-Wl,-soname,libxla_musa_mudnn_shim.so.1",
    ],
    linkshared = True,
    tags = [
        "gpu",
        "manual",
        "musa-only",
    ],
    target_compatible_with = if_musa([], ["@platforms//:incompatible"]),
    visibility = ["//visibility:public"],
    deps = [
        ":mudnn_shim_abi",
        "@local_config_musa//musa:mudnn",
        "@local_config_musa//musa:musart",
    ],
)

xla_cc_test(
    name = "mudnn_shim_abi_test",
    srcs = ["mudnn_shim_abi_test.cc"],
    deps = [
        ":mudnn_shim_abi",
        "@com_google_googletest//:gtest_main",
    ],
)
