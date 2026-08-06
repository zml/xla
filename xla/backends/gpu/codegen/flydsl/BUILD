load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tests:build_defs.bzl", "xla_test")

package(
    default_visibility = ["//xla/backends/gpu/codegen:__subpackages__"],
    licenses = ["notice"],
)

cc_library(
    name = "compiler",
    srcs = ["compiler.cc"],
    hdrs = ["compiler.h"],
    deps = [
        "@flydsl//:FlyDialect",
        "@flydsl//:FlyROCDLDialect",
        "@flydsl//:FlyToROCDL",
        "@flydsl//:FlyTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:UBDialect",
    ],
)

cc_library(
    name = "xtile_gemm",
    srcs = ["xtile_gemm.cc"],
    hdrs = ["xtile_gemm.h"],
    deps = [
        ":compiler",
        "//xla:autotuning_proto_cc",
        "//xla:status_macros",
        "//xla:xla_data_proto_cc",
        "//xla/backends/gpu/codegen/emitters:mlir_kernel_emitter",
        "//xla/backends/gpu/codegen/emitters/ir:xla_gpu",
        "//xla/codegen:ir_emission_utils",
        "//xla/codegen/emitters:kernel_api_builder",
        "//xla/codegen/emitters/ir:xla",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_query",
        "//xla/service:buffer_assignment",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:gpu_constants",
        "//xla/service/gpu:hlo_fusion_analysis",
        "//xla/service/gpu:launch_dimensions",
        "//xla/stream_executor:launch_dim",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:ROCDLDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:VectorDialect",
    ],
)

xla_cc_test(
    name = "compiler_test",
    srcs = ["compiler_test.cc"],
    deps = [
        ":compiler",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest",
        "@flydsl//:FlyDialect",
        "@flydsl//:FlyROCDLDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ROCDLDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:VectorDialect",
    ],
)

xla_test(
    name = "gemm_device_test",
    srcs = ["gemm_device_test.cc"],
    backends = ["amdgpu_any"],
    tags = [
        "gpu",
        "rocm-only",
    ],
    deps = [
        "//xla:error_spec",
        "//xla/backends/gpu/tests:hlo_pjrt_gpu_test_base",
        "//xla/tests:hlo_pjrt_interpreter_reference_mixin",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
    ],
)

sh_binary(
    name = "gemm_benchmark",
    srcs = ["gemm_benchmark.sh"],
    data = [
        "gemm_benchmark.hlo.tpl",
        "//xla/backends/gpu/autotuner:autotuner_main",
    ],
    tags = [
        "gpu",
        "manual",
        "rocm-only",
    ],
)
