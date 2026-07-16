load("//xla:xla.default.bzl", "xla_cc_binary")

package(default_visibility = ["//visibility:public"])

xla_cc_binary(
    name = "pjrt_oneapi_issue_reproducer",
    srcs = ["pjrt_oneapi_issue_reproducer.cc"],
    linkstatic = True,
    deps = [
        "//xla:literal",
        "//xla:xla_data_proto_cc",
        "//xla/backends/gpu/collectives:gpu_collectives_stub",
        "//xla/pjrt:maybe_owning_mlir_module",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt/gpu:se_gpu_pjrt_client",
        "//xla/pjrt/plugin/xla_gpu:xla_gpu_allocator_config",
        "//xla/pjrt/plugin/xla_gpu:xla_gpu_client_options",
        "//xla/service/gpu:intel_gpu_compiler",
        "//xla/stream_executor/sycl:sycl_blas_lt_plugin",
        "//xla/stream_executor/sycl:sycl_platform",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//mlir:IR",
    ],
)

xla_cc_binary(
    name = "pjrt_oneapi_reproducer",
    srcs = ["pjrt_oneapi_reproducer.cc"],
    linkstatic = True,
    deps = [
        "//xla:future",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla/hlo/builder:xla_builder",
        "//xla/backends/profiler:profiler_backends",
        "//xla/pjrt:pjrt_api",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt/c_api_client:pjrt_c_api_client",
        "//xla/python:profiler_utils",
        "//xla/service:computation_placer_hdr",
        "//xla/tsl/profiler/convert:trace_events_to_json",
        "//xla/tsl/profiler/convert:xplane_to_trace_events",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/profiler/lib:profiler_session",
        "@tsl//tsl/profiler/lib:traceme",
        "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)
