# Description:
#   Versioned host-to-vendor-LLVM protocol for the MUSA GPU backend.

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load(
    "//xla/tsl/platform:build_config.bzl",
    "tf_proto_library",
)

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
    name = "bounded_subprocess",
    srcs = ["bounded_subprocess.cc"],
    hdrs = ["bounded_subprocess.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_binary(
    name = "bounded_subprocess_test_helper",
    srcs = ["bounded_subprocess_test_helper.cc"],
)

xla_cc_test(
    name = "bounded_subprocess_test",
    srcs = ["bounded_subprocess_test.cc"],
    data = [":bounded_subprocess_test_helper"],
    deps = [
        ":bounded_subprocess",
        "//xla/tsl/platform:resource_loader",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "mcc_bundle_codegen",
    srcs = ["mcc_bundle_codegen.cc"],
    hdrs = ["mcc_bundle_codegen.h"],
    deps = [
        ":bounded_subprocess",
        ":protocol",
        "//xla/stream_executor/musa:musa_target_contract",
        "//xla/tsl/platform:env",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_binary(
    name = "mcc_bundle_codegen_test_helper",
    srcs = ["mcc_bundle_codegen_test_helper.cc"],
)

xla_cc_test(
    name = "mcc_bundle_codegen_test",
    srcs = ["mcc_bundle_codegen_test.cc"],
    data = [":mcc_bundle_codegen_test_helper"],
    deps = [
        ":mcc_bundle_codegen",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:resource_loader",
        "//xla/tsl/testing:temporary_directory",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
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

cc_library(
    name = "musa_compilation_provider",
    srcs = ["musa_compilation_provider.cc"],
    hdrs = ["musa_compilation_provider.h"],
    deps = [
        ":bounded_subprocess",
        ":musa_bridge_ir_validator",
        ":musa_llvm14_compatibility",
        ":musa_shim_abi",
        ":protocol",
        ":protocol_proto_cc",
        "//xla/stream_executor/musa:musa_mubin",
        "//xla/tsl/platform:env",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "musa_executable_envelope",
    srcs = ["musa_executable_envelope.cc"],
    hdrs = ["musa_executable_envelope.h"],
    deps = [
        ":musa_compilation_provider",
        ":musa_llvm14_compatibility",
        ":musa_shim_abi",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:semantic_version",
        "//xla/stream_executor/abi:executable_abi_version",
        "//xla/stream_executor/abi:executable_abi_version_proto_cc",
        "//xla/stream_executor/musa:musa_executable_abi",
        "//xla/stream_executor/musa:musa_mubin",
        "//xla/stream_executor/musa:musa_platform_id",
        "//xla/stream_executor/musa:musa_target_contract",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "musa_executable_envelope_test",
    srcs = ["musa_executable_envelope_test.cc"],
    deps = [
        ":musa_compilation_provider",
        ":musa_executable_envelope",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:semantic_version",
        "//xla/stream_executor/abi:executable_abi_version",
        "//xla/stream_executor/musa:musa_compute_capability",
        "//xla/stream_executor/musa:musa_executable_abi",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "musa_compiler_bundle",
    srcs = ["musa_compiler_bundle.cc"],
    hdrs = ["musa_compiler_bundle.h"],
    linkopts = ["-ldl"],
    deps = [
        ":musa_compilation_provider",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "musa_compiler_bundle_test",
    srcs = ["musa_compiler_bundle_test.cc"],
    deps = [
        ":musa_compilation_provider",
        ":musa_compiler_bundle",
        "//xla/tsl/platform:env",
        "//xla/tsl/testing:temporary_directory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_binary(
    name = "musa_compilation_provider_test_helper",
    testonly = True,
    srcs = ["musa_compilation_provider_test_helper.cc"],
    deps = [
        ":protocol",
        ":protocol_proto_cc",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "musa_compilation_provider_test",
    srcs = ["musa_compilation_provider_test.cc"],
    data = [":musa_compilation_provider_test_helper"],
    deps = [
        ":musa_compilation_provider",
        ":musa_llvm14_compatibility",
        ":musa_shim_abi",
        ":protocol",
        "//xla/stream_executor/musa:musa_mubin",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:resource_loader",
        "//xla/tsl/testing:temporary_directory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "musa_llvm14_compatibility",
    srcs = ["musa_llvm14_compatibility.cc"],
    hdrs = ["musa_llvm14_compatibility.h"],
    deps = [
        ":musa_bridge_ir_validator",
        ":musa_shim_abi",
        ":protocol",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:AsmParser",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TransformUtils",
    ],
)

filegroup(
    name = "musa_llvm14_compatibility_goldens",
    srcs = glob(["testdata/llvm14_compatibility/*.ll"]),
)

xla_cc_test(
    name = "musa_llvm14_compatibility_test",
    srcs = ["musa_llvm14_compatibility_test.cc"],
    data = [":musa_llvm14_compatibility_goldens"],
    deps = [
        ":musa_llvm14_compatibility",
        ":musa_shim_abi",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
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
        "@boringssl//:crypto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_protobuf//:protobuf",
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
