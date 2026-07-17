# Description:
#   Vendor-LLVM side of the isolated MUSA compiler bridge.

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla:friends"]),
    licenses = ["notice"],
)

cc_library(
    name = "toolchain_fingerprint",
    srcs = ["toolchain_fingerprint.cc"],
    hdrs = ["toolchain_fingerprint.h"],
    deps = [
        "//xla/service/gpu/musa:musa_shim_abi",
        "//xla/service/gpu/musa:protocol",
        "@boringssl//:crypto",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "toolchain_fingerprint_test",
    srcs = ["toolchain_fingerprint_test.cc"],
    deps = [
        ":toolchain_fingerprint",
        "//xla/service/gpu/musa:protocol",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_binary(
    name = "musa-llvm-bridge",
    srcs = ["main.cc"],
    tags = [
        "gpu",
        "musa-only",
    ],
    target_compatible_with = select({
        "@local_config_musa//musa:using_musa": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":bridge_core",
        ":toolchain_fingerprint",
        "//xla/service/gpu/musa:mcc_bundle_codegen",
        "//xla/service/gpu/musa:protocol",
        "//xla/service/gpu/musa:protocol_proto_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "main_process_test",
    size = "medium",
    srcs = ["main_process_test.cc"],
    data = [
        "testdata/minimal.ll",
        ":musa-llvm-bridge",
        "@local_config_musa//musa:clang_offload_bundler",
        "@local_config_musa//musa:libclang_cpp_file",
        "@local_config_musa//musa:libdevice",
        "@local_config_musa//musa:lld",
        "@local_config_musa//musa:llvm_readobj",
        "@local_config_musa//musa:mcc",
        "@local_config_musa//musa:toolchain_data",
        "@local_config_musa//musa:toolchain_identity",
    ],
    env = {
        "MUSA_TEST_BUNDLER": "$(rlocationpath @local_config_musa//musa:clang_offload_bundler)",
        "MUSA_TEST_IDENTITY": "$(rlocationpath @local_config_musa//musa:toolchain_identity)",
        "MUSA_TEST_LIBDEVICE": "$(rlocationpath @local_config_musa//musa:libdevice)",
        "MUSA_TEST_LIBCLANG_CPP": "$(rlocationpath @local_config_musa//musa:libclang_cpp_file)",
        "MUSA_TEST_LLD": "$(rlocationpath @local_config_musa//musa:lld)",
        "MUSA_TEST_MCC": "$(rlocationpath @local_config_musa//musa:mcc)",
        "MUSA_TEST_READOBJ": "$(rlocationpath @local_config_musa//musa:llvm_readobj)",
    },
    tags = [
        "gpu",
        "musa-only",
    ],
    target_compatible_with = select({
        "@local_config_musa//musa:using_musa": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":toolchain_fingerprint",
        "//xla/service/gpu/musa:mcc_bundle_codegen",
        "//xla/service/gpu/musa:musa_shim_abi",
        "//xla/service/gpu/musa:protocol",
        "//xla/service/gpu/musa:protocol_proto_cc",
        "//xla/tsl/platform:resource_loader",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "bridge_core",
    srcs = [
        "bridge_core.cc",
        "mubin_validator.cc",
    ],
    hdrs = [
        "bridge_core.h",
        "mubin_validator.h",
    ],
    # The MUSA fork conditionally exposes target enum values and changes the
    # vendor LLVM header ABI under this define. It must match libclang-cpp.so.14.
    defines = ["__MTGPU__"],
    linkopts = ["-ldl"],
    tags = [
        "gpu",
        "musa-only",
    ],
    target_compatible_with = select({
        "@local_config_musa//musa:using_musa": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        "//xla/service/gpu/musa:musa_shim_abi",
        "//xla/service/gpu/musa:protocol",
        "//xla/service/gpu/musa:protocol_proto_cc",
        "//xla/stream_executor/musa:musa_mubin",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@local_config_musa//musa:musa_llvm14",
    ],
)

cc_test(
    name = "bridge_core_test",
    size = "small",
    srcs = ["bridge_core_test.cc"],
    data = glob(["testdata/*.ll"]) + [
        "//xla/service/gpu/musa:musa_llvm14_compatibility_goldens",
    ],
    tags = [
        "gpu",
        "musa-only",
    ],
    target_compatible_with = select({
        "@local_config_musa//musa:using_musa": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":bridge_core",
        "//xla/service/gpu/musa:musa_shim_abi",
        "//xla/service/gpu/musa:protocol",
        "//xla/service/gpu/musa:protocol_proto_cc",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

xla_cc_test(
    name = "bridge_live_test",
    size = "large",
    srcs = ["bridge_live_test.cc"],
    data = [
        ":musa-llvm-bridge",
        "@local_config_musa//musa:clang_offload_bundler",
        "@local_config_musa//musa:libclang_cpp_file",
        "@local_config_musa//musa:libdevice",
        "@local_config_musa//musa:lld",
        "@local_config_musa//musa:llvm_readobj",
        "@local_config_musa//musa:mcc",
        "@local_config_musa//musa:toolchain_data",
        "@local_config_musa//musa:toolchain_identity",
    ],
    env = {
        "MUSA_BRIDGE_RUNFILE": "$(rlocationpath :musa-llvm-bridge)",
        "MUSA_BUNDLER_RUNFILE": "$(rlocationpath @local_config_musa//musa:clang_offload_bundler)",
        "MUSA_LIBDEVICE_RUNFILE": "$(rlocationpath @local_config_musa//musa:libdevice)",
        "MUSA_LIBCLANG_RUNFILE": "$(rlocationpath @local_config_musa//musa:libclang_cpp_file)",
        "MUSA_LLD_RUNFILE": "$(rlocationpath @local_config_musa//musa:lld)",
        "MUSA_MCC_RUNFILE": "$(rlocationpath @local_config_musa//musa:mcc)",
        "MUSA_READOBJ_RUNFILE": "$(rlocationpath @local_config_musa//musa:llvm_readobj)",
        "MUSA_TOOLCHAIN_IDENTITY_RUNFILE": "$(rlocationpath @local_config_musa//musa:toolchain_identity)",
    },
    linkstatic = True,
    tags = [
        "gpu",
        "manual",
        "musa-only",
        "no_oss",
    ],
    target_compatible_with = select({
        "@local_config_musa//musa:using_musa": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        ":toolchain_fingerprint",
        "//xla/service/gpu/musa:mcc_bundle_codegen",
        "//xla/service/gpu/musa:musa_shim_abi",
        "//xla/service/gpu/musa:protocol",
        "//xla/service/gpu/musa:protocol_proto_cc",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:kernel",
        "//xla/stream_executor:kernel_args",
        "//xla/stream_executor:kernel_spec",
        "//xla/stream_executor:launch_dim",
        "//xla/stream_executor:module_spec",
        "//xla/stream_executor:stream",
        "//xla/stream_executor/musa:musa_executor",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "@bazel_tools//tools/cpp/runfiles",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)
