# Description:
#   Versioned host-to-vendor-LLVM protocol for the MUSA GPU backend.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")
load(
    "//xla/tsl/platform:build_config.bzl",
    "tf_proto_library",
)
load("//xla/tsl:tsl.bzl", "internal_visibility")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = ["//xla:friends"],
)

cc_library(
    name = "musa_shim_abi",
    srcs = ["musa_shim_abi.cc"],
    hdrs = [
        "musa_shim_abi.def",
        "musa_shim_abi.h",
    ],
    deps = [
        "//xla/stream_executor/musa:musa_target_contract",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "musa_shim_abi_test",
    srcs = ["musa_shim_abi_test.cc"],
    deps = [
        ":musa_shim_abi",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "musa_bridge_ir_validator",
    srcs = ["musa_bridge_ir_validator.cc"],
    hdrs = ["musa_bridge_ir_validator.h"],
    deps = [
        ":musa_shim_abi",
        ":protocol",
        "//xla/stream_executor/musa:musa_target_contract",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:AsmParser",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
    ],
)

xla_cc_test(
    name = "musa_bridge_ir_validator_test",
    srcs = ["musa_bridge_ir_validator_test.cc"],
    deps = [
        ":musa_bridge_ir_validator",
        ":musa_shim_abi",
        ":protocol",
        ":protocol_proto_cc",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_proto_library(
    name = "protocol_proto",
    srcs = ["protocol.proto"],
    make_default_target_header_only = True,
)

cc_library(
    name = "protocol",
    srcs = ["protocol.cc"],
    hdrs = ["protocol.h"],
    deps = [
        ":musa_shim_abi",
        ":protocol_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_protobuf//:protobuf",
        "@boringssl//:crypto",
    ],
)

xla_cc_test(
    name = "protocol_test",
    srcs = ["protocol_test.cc"],
    deps = [
        ":protocol",
        ":protocol_proto_cc",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
    ],
)
