# Description:
#   Apple Metal device tracer for the XLA/TSL profiler (xprof). Links into
#   libpjrt_metal.dylib via //xla/backends/profiler/gpu:device_tracer (the
#   if_macos arm) so its RegisterProfilerFactory static initializer fires at
#   .dylib load and the existing ZML `--profile` path collects a GPU plane.

load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla:internal"]),
    licenses = ["notice"],
)

cc_library(
    name = "device_tracer_metal",
    srcs = ["device_tracer_metal.cc"],
    tags = [
        "gpu",
        "manual",
        "metal-only",
    ],
    deps = [
        "//xla/stream_executor/metal:metal_runtime",
        "//xla/tsl/profiler/utils:time_utils",
        "//xla/tsl/profiler/utils:xplane_builder",
        "//xla/tsl/profiler/utils:xplane_schema",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/profiler/lib:profiler_factory",
        "@tsl//tsl/profiler/lib:profiler_interface",
        "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
    alwayslink = True,
)
