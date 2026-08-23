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
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ROCDLDialect",
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

cc_library(
    name = "xtile_softmax",
    srcs = ["xtile_softmax.cc"],
    hdrs = ["xtile_softmax.h"],
    visibility = [
        "//xla/backends/gpu/autotuner:__pkg__",
        "//xla/backends/gpu/codegen:__subpackages__",
    ],
    deps = [
        ":compiler",
        "//xla:status_macros",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/gpu/codegen/emitters:mlir_kernel_emitter",
        "//xla/backends/gpu/codegen/emitters/ir:xla_gpu",
        "//xla/codegen:ir_emission_utils",
        "//xla/codegen/emitters:kernel_api_builder",
        "//xla/hlo/ir:hlo",
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
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:ROCDLDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:TensorDialect",
    ],
)

cc_library(
    name = "xtile_transpose",
    srcs = ["xtile_transpose.cc"],
    hdrs = ["xtile_transpose.h"],
    visibility = [
        "//xla/backends/gpu/autotuner:__pkg__",
        "//xla/backends/gpu/codegen:__subpackages__",
    ],
    deps = [
        ":compiler",
        "//xla:status_macros",
        "//xla:xla_data_proto_cc",
        "//xla/backends/gpu/codegen/emitters:mlir_kernel_emitter",
        "//xla/backends/gpu/codegen/emitters/ir:xla_gpu",
        "//xla/codegen:ir_emission_utils",
        "//xla/codegen/emitters:kernel_api_builder",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/service/gpu:gpu_constants",
        "//xla/service/gpu:hlo_fusion_analysis",
        "//xla/service/gpu:launch_dimensions",
        "//xla/stream_executor:launch_dim",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:VectorDialect",
    ],
)

xla_cc_test(
    name = "xtile_softmax_test",
    srcs = ["xtile_softmax_test.cc"],
    deps = [
        ":xtile_softmax",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/service/gpu:hlo_fusion_analysis",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

xla_cc_test(
    name = "xtile_transpose_test",
    srcs = ["xtile_transpose_test.cc"],
    deps = [
        ":xtile_transpose",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/service/gpu:hlo_fusion_analysis",
        "//xla/tests:xla_internal_test_main",
        "@com_google_googletest//:gtest",
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
    env = {
        "HIPBLASLT_PRELOAD_KERNELS": "0",
        "HIPBLASLT_ROCROLLER_NO_CUSTOM_KERNEL": "1",
    },
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

xla_test(
    name = "fusion_device_test",
    srcs = ["fusion_device_test.cc"],
    backends = ["amdgpu_any"],
    env = {
        "HIPBLASLT_PRELOAD_KERNELS": "0",
        "HIPBLASLT_ROCROLLER_NO_CUSTOM_KERNEL": "1",
    },
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
        "batched_gemm_benchmark.hlo.tpl",
        "bitcast_input_gemm_benchmark.hlo.tpl",
        "chain_gemm_benchmark.hlo.tpl",
        "concat_input_gemm_benchmark.hlo.tpl",
        "converted_input_gemm_benchmark.hlo.tpl",
        "dynamic_slice_input_gemm_benchmark.hlo.tpl",
        "gemm_benchmark.hlo.tpl",
        "narrowing_gemm_benchmark.hlo.tpl",
        "relu_gemm_benchmark.hlo.tpl",
        "scalar_gemm_benchmark.hlo.tpl",
        "scaled_gemm_benchmark.hlo.tpl",
        "slice_input_gemm_benchmark.hlo.tpl",
        "vector_gemm_benchmark.hlo.tpl",
        "//xla/backends/gpu/autotuner:autotuner_main",
    ],
    tags = [
        "gpu",
        "manual",
        "rocm-only",
    ],
)

sh_binary(
    name = "fusion_benchmark",
    srcs = ["fusion_benchmark.sh"],
    data = [
        "fly_fusion_bf16_row_reduction_benchmark.hlo",
        "fly_fusion_bf16_softmax_benchmark.hlo",
        "fly_fusion_bf16_transpose_1024_benchmark.hlo",
        "fly_fusion_bf16_transpose_4096_benchmark.hlo",
        "fly_fusion_bf16_transpose_4096x16384_benchmark.hlo",
        "fly_fusion_bf16_transpose_benchmark.hlo",
        "fly_fusion_elementwise_hbm_benchmark.hlo",
        "fly_fusion_row_reduction_benchmark.hlo",
        "//xla/backends/gpu/autotuner:autotuner_main",
    ],
    tags = [
        "gpu",
        "manual",
        "rocm-only",
    ],
)
