load("//xla/tsl/platform:rules_cc.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "vulkan_platform_id",
    srcs = ["vulkan_platform_id.cc"],
    hdrs = ["vulkan_platform_id.h"],
    deps = ["//xla/stream_executor:platform"],
)

cc_library(
    name = "vulkan_platform",
    srcs = ["vulkan_platform.cc"],
    hdrs = ["vulkan_platform.h"],
    deps = [
        ":vulkan_platform_id",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/platform:initialize",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
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


