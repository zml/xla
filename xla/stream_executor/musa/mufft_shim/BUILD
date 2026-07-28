"""Versioned, vendor-linked adapter for the MUSA muFFT library."""

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
    name = "mufft_shim_abi",
    hdrs = ["mufft_shim_abi.h"],
    visibility = ["//visibility:public"],
)

# The PJRT plugin discovers this DSO next to itself at runtime. The vendor SDK
# is intentionally confined to this target and must never become a plugin link
# dependency.
cc_binary(
    name = "libxla_musa_mufft_shim.so",
    srcs = ["mufft_shim.cc"],
    additional_linker_inputs = ["mufft_shim_version_script.lds"],
    copts = ["-fvisibility=hidden"],
    linkopts = [
        "-Wl,--no-undefined",
        "-Wl,--version-script,$(location mufft_shim_version_script.lds)",
        "-Wl,-rpath,$$ORIGIN",
        "-Wl,-soname,libxla_musa_mufft_shim.so.1",
    ],
    linkshared = True,
    tags = [
        "gpu",
        "musa-only",
    ],
    target_compatible_with = if_musa([], ["@platforms//:incompatible"]),
    visibility = ["//visibility:public"],
    deps = [
        ":mufft_shim_abi",
        "@local_config_musa//musa:mufft",
    ],
)

xla_cc_test(
    name = "mufft_shim_abi_test",
    srcs = ["mufft_shim_abi_test.cc"],
    deps = [
        ":mufft_shim_abi",
        "@com_google_googletest//:gtest_main",
    ],
)
