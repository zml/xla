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

# A first-class GpuComputeCapability alternative for Metal (peer to the
# CUDA/ROCm/oneAPI capability types). Header-only, pure C++ (no Metal SDK) so it
# links into the core device_description on every platform, like the others.
cc_library(
    name = "metal_compute_capability",
    hdrs = ["metal_compute_capability.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/stream_executor:device_description_proto_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "metal_runtime",
    # metal_runtime.cc is Objective-C++. We force the language with clang's
    # "-ObjC++" driver flag rather than "-x objective-c++": Bazel places copts
    # AFTER the source on this (hermetic) toolchain's command line, and clang's
    # positional -x only affects inputs that FOLLOW it, so "-x objective-c++"
    # would be a no-op and the file would miscompile as plain C++. "-ObjC++" is
    # non-positional (handled once in the driver) so it works regardless of
    # ordering. cc_library also refuses ".mm" sources, so renaming is not an
    # option here. See metal-xla-docs/04-VERIFICATION-FINDINGS.md.
    srcs = ["metal_runtime.cc"],
    hdrs = ["metal_runtime.h"],
    # Compile as Objective-C++ (Metal/Foundation headers). Use clang's
    # non-positional "-ObjC++" driver flag, NOT "-x objective-c++": Bazel places
    # copts after the source on this toolchain and clang's positional -x would
    # then be a no-op (the file would miscompile as plain C++). cc_library also
    # refuses ".mm" sources. We deliberately do NOT pass "-isysroot" to a system
    # SDK: the hermetic macOS toolchain already supplies a full macOS SDK (which
    # contains Metal.framework + Foundation.framework), and overriding it with an
    # absolute CommandLineTools path trips Bazel's hermetic include validation
    # ("absolute path inclusion" / "outside the execution root"). See
    # metal-xla-docs/04-VERIFICATION-FINDINGS.md.
    copts = tsl_copts() + if_macos([
        "-ObjC++",
        "-fobjc-arc",
    ]),
    linkopts = if_macos([
        "-Wl,-framework,Foundation",
        # IOKit: the GPU core count has no Metal API; it is read from the
        # IORegistry "gpu-core-count" property (see QueryGpuCoreCount).
        "-Wl,-framework,IOKit",
        "-Wl,-framework,Metal",
        "-Wl,-framework,MetalPerformanceShaders",
    ]),
    deps = [
        "//xla/stream_executor:launch_dim",
        "//xla/tsl/platform:status_macros",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
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
        ":metal_stream",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:kernel_args",
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
        "//xla/stream_executor:kernel_args",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:platform",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_common",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "metal_executor_hdrs",
    hdrs = ["metal_executor.h"],
    deps = [
        ":metal_runtime",
        "//xla/stream_executor:blas",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:device_description",
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
        ":metal_compute_capability",
        ":metal_event",
        ":metal_executor_hdrs",
        ":metal_kernel",
        ":metal_runtime",
        ":metal_stream",
        "//xla/stream_executor:blas",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:device_description",
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

# The Metal peer of cuda:all_runtime / rocm:all_runtime: the single label a
# binary deps on to pull in the Metal platform's registration. It has no srcs of
# its own -- all it does is force-link :metal_platform, whose static initializer
# registers the platform. `alwayslink` is therefore LOAD-BEARING: without it the
# linker drops the initializer and the platform silently never registers.
#
# It registers no kernels. (This note used to describe a TopKKernel<K,KT,VT>
# registration peer to cuda:topk_kernel_cuda; no such thing exists here or ever
# did. Metal's TopK is a thunk -- //xla/backends/gpu/runtime:metal_topk_thunk --
# not a stream_executor custom kernel.)
cc_library(
    name = "all_runtime",
    visibility = ["//visibility:public"],
    deps = if_macos([
        ":metal_platform",
    ]),
    alwayslink = True,
)
