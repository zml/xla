# CUTLASS 4.6.1 — vendored separately from cutlass 3.8 for sm120 block-scaled fp4 (RTX 5090).
# Bumped 4.4.2 -> 4.6.1 to unlock FlashInfer's thin-N block-scaled TMA tiles
# (128x32/64 x K256), which 4.4.2 rejected via an SF-TMA size-equivalence assert.
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD-3

exports_files(["LICENSE.txt"])

filegroup(
    name = "cutlass4_header_files",
    srcs = glob(["include/**"]),
)

filegroup(
    name = "cutlass4_util_header_files",
    srcs = glob(["tools/util/include/**"]),
)

cc_library(
    name = "cutlass4",
    hdrs = [
        ":cutlass4_header_files",
        ":cutlass4_util_header_files",
    ],
    includes = [
        "include",
        "tools/util/include",
    ],
)
