"""MUSA-platform specific StreamExecutor support code."""

load(
    "@local_config_musa//musa:build_defs.bzl",
    "if_musa",
)
load(
    "//xla/stream_executor:build_defs.bzl",
    "stream_executor_friends",
)
load("//xla/tsl:tsl.bzl", "internal_visibility", "tsl_copts")
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
    name = "musa_compute_capability",
    hdrs = ["musa_compute_capability.h"],
    deps = [
        "//xla/stream_executor:device_description_proto_cc",
    ],
)

cc_library(
    name = "musa_platform_id",
    srcs = ["musa_platform_id.cc"],
    hdrs = ["musa_platform_id.h"],
    deps = ["//xla/stream_executor:platform_id"],
)

cc_library(
    name = "musa_status",
    srcs = ["musa_status.cc"],
    hdrs = ["musa_status.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "musa_runtime",
    srcs = ["musa_runtime.cc"],
    hdrs = ["musa_runtime.h"],
    linkopts = ["-ldl"],
    tags = [
        "gpu",
        "musa-only",
    ],
    deps = [
        ":musa_status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@local_config_musa//musa:musa_driver",
        "@local_config_musa//musa:musart",
    ],
)

cc_library(
    name = "musa_event",
    srcs = ["musa_event.cc"],
    hdrs = ["musa_event.h"],
    tags = [
        "gpu",
        "musa-only",
    ],
    deps = [
        ":musa_runtime",
        "//xla/stream_executor:event",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "musa_stream",
    srcs = ["musa_stream.cc"],
    hdrs = ["musa_stream.h"],
    tags = [
        "gpu",
        "musa-only",
    ],
    deps = [
        ":musa_event",
        ":musa_runtime",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:event",
        "//xla/stream_executor:event_based_timer",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_common",
        "//xla/stream_executor:stream_executor_h",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "musa_runtime_abi_version",
    srcs = ["musa_runtime_abi_version.cc"],
    hdrs = ["musa_runtime_abi_version.h"],
    tags = [
        "gpu",
        "musa-only",
    ],
    deps = [
        ":musa_platform_id",
        ":musa_runtime",
        "//xla/stream_executor:platform_id",
        "//xla/stream_executor:semantic_version",
        "//xla/stream_executor/abi:executable_abi_version",
        "//xla/stream_executor/abi:runtime_abi_version",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "musa_executor",
    srcs = ["musa_executor.cc"],
    hdrs = ["musa_executor.h"],
    tags = [
        "gpu",
        "musa-only",
    ],
    deps = [
        ":musa_compute_capability",
        ":musa_event",
        ":musa_platform_id",
        ":musa_runtime",
        ":musa_stream",
        "//xla/stream_executor:activate_context",
        "//xla/stream_executor:blas",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:dnn",
        "//xla/stream_executor:event",
        "//xla/stream_executor:event_based_timer",
        "//xla/stream_executor:fft",
        "//xla/stream_executor:generic_memory_allocation",
        "//xla/stream_executor:generic_memory_allocator",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:memory_allocation",
        "//xla/stream_executor:memory_space",
        "//xla/stream_executor:module_spec",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:semantic_version",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "//xla/tsl/platform:status_macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_config_musa//musa:mublas",
        "@local_config_musa//musa:mudnn",
        "@local_config_musa//musa:musa_driver",
        "@local_config_musa//musa:musart",
    ],
    alwayslink = True,
)

cc_library(
    name = "musa_platform",
    srcs = ["musa_platform.cc"],
    hdrs = ["musa_platform.h"],
    tags = [
        "gpu",
        "musa-only",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":musa_executor",
        ":musa_platform_id",
        ":musa_runtime",
        ":musa_runtime_abi_version",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:executor_cache",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/platform:initialize",
        "//xla/tsl/platform:status_macros",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = True,
)

cc_library(
    name = "all_runtime",
    copts = tsl_copts(),
    visibility = ["//visibility:public"],
    deps = if_musa([
        ":musa_platform",
    ]),
    alwayslink = True,
)
