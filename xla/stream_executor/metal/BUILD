# Description:
#   Metal-platform specific StreamExecutor support code.

load("//xla/stream_executor:build_defs.bzl", "stream_executor_friends")
load("//xla/tsl:tsl.bzl", "if_macos", "internal_visibility", "tsl_copts")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = stream_executor_friends(),
)

cc_library(
    name = "metal_platform_id",
    srcs = ["metal_platform_id.cc"],
    hdrs = ["metal_platform_id.h"],
    visibility = ["//visibility:public"],
    deps = ["//xla/stream_executor:platform_id"],
)

cc_library(
    name = "metal_runtime",
    srcs = ["metal_runtime.cc"],
    hdrs = ["metal_runtime.h"],
    copts = tsl_copts() + if_macos([
        "-x",
        "objective-c++",
        "-fobjc-arc",
        "-isysroot",
        "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk",
    ]),
    linkopts = if_macos([
        "-Wl,-framework,Foundation",
        "-Wl,-framework,Metal",
    ]),
    deps = [
        "//xla/stream_executor:event",
        "//xla/stream_executor:launch_dim",
        "//xla/tsl/platform:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "metal_event",
    srcs = ["metal_event.cc"],
    hdrs = ["metal_event.h"],
    deps = [
        ":metal_runtime",
        "//xla/stream_executor:event",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "metal_kernel",
    srcs = ["metal_kernel.cc"],
    hdrs = ["metal_kernel.h"],
    deps = [
        ":metal_runtime",
        "//xla/stream_executor:kernel_args",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "metal_stream",
    srcs = ["metal_stream.cc"],
    hdrs = ["metal_stream.h"],
    deps = [
        ":metal_event",
        ":metal_executor_hdrs",
        ":metal_runtime",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:event",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_common",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "metal_executor_hdrs",
    hdrs = ["metal_executor.h"],
    deps = [
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:blas",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:event",
        "//xla/stream_executor:fft",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:memory_allocation",
        "//xla/stream_executor:memory_allocator",
        "//xla/stream_executor:memory_space",
        "//xla/stream_executor:module_spec",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "metal_executor",
    srcs = ["metal_executor.cc"],
    deps = [
        ":metal_event",
        ":metal_executor_hdrs",
        ":metal_kernel",
        ":metal_runtime",
        ":metal_stream",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:blas",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:event",
        "//xla/stream_executor:fft",
        "//xla/stream_executor:generic_memory_allocation",
        "//xla/stream_executor:generic_memory_allocator",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:memory_allocation",
        "//xla/stream_executor:memory_allocator",
        "//xla/stream_executor:memory_space",
        "//xla/stream_executor:module_spec",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:semantic_version",
        "//xla/stream_executor:stream",
        "//xla/tsl/platform:status_macros",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "metal_platform",
    srcs = ["metal_platform.cc"],
    hdrs = ["metal_platform.h"],
    tags = [
        "gpu",
        "metal-only",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":metal_executor",
        ":metal_platform_id",
        ":metal_runtime",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:executor_cache",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/platform:initialize",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = True,
)

cc_library(
    name = "all_runtime",
    visibility = ["//visibility:public"],
    deps = if_macos([
        ":metal_platform",
    ]),
    alwayslink = True,
)
