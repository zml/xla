load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "vulkan_platform_id",
    srcs = ["vulkan_platform_id.cc"],
    hdrs = ["vulkan_platform_id.h"],
    deps = ["//xla/stream_executor:platform"],
)

cc_library(
    name = "vulkan_executor",
    srcs = ["vulkan_executor.cc"],
    hdrs = ["vulkan_executor.h"],
    linkopts = ["-ldl"],
    deps = [
        "//xla/stream_executor:activate_context",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:event",
        "//xla/stream_executor:generic_memory_allocation",
        "//xla/stream_executor:generic_memory_allocator",
        "//xla/stream_executor/gpu:gpu_executor_header",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:kernel_args",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:memory_allocation",
        "//xla/stream_executor:memory_allocator",
        "//xla/stream_executor:memory_space",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_common",
        "//xla/stream_executor:stream_executor_h",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:status_macros",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@vulkan_headers//:vulkan_headers_no_prototypes",
    ],
)

cc_library(
    name = "vulkan_platform",
    srcs = ["vulkan_platform.cc"],
    hdrs = ["vulkan_platform.h"],
    deps = [
        ":vulkan_executor",
        ":vulkan_platform_id",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:executor_cache",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/platform:initialize",
        "//xla/tsl/platform:status_macros",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = True,
)

xla_cc_test(
    name = "vulkan_platform_test",
    srcs = ["vulkan_platform_test.cc"],
    deps = [
        ":vulkan_platform",
        ":vulkan_platform_id",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/tsl/platform:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)
