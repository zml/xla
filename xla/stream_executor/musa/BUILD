"""MUSA-platform specific StreamExecutor support code."""

load(
    "@local_config_musa//musa:build_defs.bzl",
    "if_musa",
    "musa_config",
)
load(
    "//xla/stream_executor:build_defs.bzl",
    "stream_executor_friends",
)
load("//xla:xla.default.bzl", "xla_cc_test")
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
    name = "musa_dso_loader",
    srcs = ["musa_dso_loader.cc"],
    hdrs = ["musa_dso_loader.h"],
    linkopts = ["-ldl"],
    tags = [
        "gpu",
        "musa-only",
    ],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "musa_dso_loader_test",
    srcs = ["musa_dso_loader_test.cc"],
    tags = [
        "gpu",
        "musa-only",
    ],
    deps = [
        ":musa_dso_loader",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "musa_driver",
    srcs = ["musa_driver.cc"],
    hdrs = ["musa_driver.h"],
    tags = [
        "gpu",
        "musa-only",
    ],
    deps = [
        ":musa_dso_loader",
        ":musa_status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_config_musa//musa:musa_headers",
    ],
)

cc_library(
    name = "musa_context",
    srcs = ["musa_context.cc"],
    hdrs = ["musa_context.h"],
    tags = [
        "gpu",
        "musa-only",
    ],
    deps = [
        ":musa_driver",
        "//xla/stream_executor/gpu:context",
        "//xla/stream_executor/gpu:scoped_activate_context",
        "//xla/tsl/platform:status_macros",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

xla_cc_test(
    name = "musa_driver_test",
    srcs = ["musa_driver_test.cc"],
    tags = [
        "gpu",
        "musa-only",
    ],
    deps = [
        ":musa_driver",
        ":musa_dso_loader",
        ":musa_status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_config_musa//musa:musa_headers",
    ],
)

cc_library(
    name = "musa_compute_capability",
    hdrs = ["musa_compute_capability.h"],
    deps = [
        "//xla/stream_executor:device_description_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

xla_cc_test(
    name = "musa_compute_capability_test",
    srcs = ["musa_compute_capability_test.cc"],
    deps = [
        ":musa_compute_capability",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "musa_platform_id",
    srcs = ["musa_platform_id.cc"],
    hdrs = ["musa_platform_id.h"],
    deps = ["//xla/stream_executor:platform_id"],
)

cc_library(
    name = "musa_device_properties",
    hdrs = ["musa_device_properties.h"],
)

cc_library(
    name = "musa_target_contract",
    hdrs = ["musa_target_contract.h"],
)

cc_library(
    name = "musa_device_description",
    srcs = ["musa_device_description.cc"],
    hdrs = ["musa_device_description.h"],
    deps = [
        ":musa_compute_capability",
        ":musa_device_properties",
        ":musa_target_contract",
        ":musa_version_parser",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:semantic_version",
        "//xla/tsl/platform:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

xla_cc_test(
    name = "musa_device_description_test",
    srcs = ["musa_device_description_test.cc"],
    deps = [
        ":musa_device_description",
        ":musa_device_properties",
        ":musa_target_contract",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:memory_allocation",
        "//xla/stream_executor:memory_allocator",
        "//xla/stream_executor:memory_space",
        "//xla/stream_executor:semantic_version",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "musa_status",
    srcs = ["musa_status.cc"],
    hdrs = ["musa_status.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@local_config_musa//musa:musa_headers",
    ],
)

cc_library(
    name = "musa_version_parser",
    srcs = ["musa_version_parser.cc"],
    hdrs = ["musa_version_parser.h"],
    deps = [
        "//xla/stream_executor:semantic_version",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "musa_version_parser_test",
    srcs = ["musa_version_parser_test.cc"],
    deps = [
        ":musa_version_parser",
        "//xla/stream_executor:semantic_version",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "musa_runtime",
    srcs = ["musa_runtime.cc"],
    hdrs = ["musa_runtime.h"],
    tags = [
        "gpu",
        "musa-only",
    ],
    deps = [
        ":musa_dso_loader",
        ":musa_device_properties",
        ":musa_status",
        "//xla/tsl/platform:status_macros",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_config_musa//musa:musa_headers",
    ],
)

xla_cc_test(
    name = "musa_runtime_test",
    srcs = ["musa_runtime_test.cc"],
    tags = [
        "gpu",
        "musa-only",
    ],
    deps = [
        ":musa_dso_loader",
        ":musa_runtime",
        ":musa_status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_config_musa//musa:musa_headers",
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
        "//xla/stream_executor:activate_context",
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
        "//xla/stream_executor:activate_context",
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
    local_defines = [
        "XLA_MUSA_TOOLKIT_VERSION=%s" % musa_config().version_number,
    ],
    deps = [
        ":musa_context",
        ":musa_device_description",
        ":musa_device_properties",
        ":musa_driver",
        ":musa_event",
        ":musa_platform_id",
        ":musa_runtime",
        ":musa_stream",
        ":musa_version_parser",
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
        "//xla/stream_executor/gpu:scoped_activate_context",
        "//xla/tsl/platform:status_macros",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
    ],
    alwayslink = True,
)

xla_cc_test(
    name = "musa_executor_test",
    srcs = ["musa_executor_test.cc"],
    tags = [
        "gpu",
        "musa-only",
    ],
    deps = [
        ":musa_compute_capability",
        ":musa_driver",
        ":musa_executor",
        "//xla/stream_executor:activate_context",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:semantic_version",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "musa_context_test",
    srcs = ["musa_context_test.cc"],
    tags = [
        "gpu",
        "musa-only",
    ],
    deps = [
        ":musa_context",
        ":musa_driver",
        "//xla/stream_executor/gpu:scoped_activate_context",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@local_config_musa//musa:musa_headers",
    ],
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
        ":musa_version_parser",
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
