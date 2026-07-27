load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

package(default_visibility = ["//visibility:public"])

cuda_library(
    name = "cutlass_fp4_mm_lib",
    srcs = ["cutlass_fp4_mm.cu.cc"],
    tags = ["cuda-only", "gpu"],
    deps = [
        "@cutlass4_archive//:cutlass4",
        "@local_config_cuda//cuda:cuda_headers",
    ],
    alwayslink = True,
)

cc_binary(
    name = "cutlass_fp4_mm",
    tags = ["cuda-only", "gpu"],
    deps = [
        ":cutlass_fp4_mm_lib",
        "@local_config_cuda//cuda:cudart",
    ],
)

# Production FlashInfer runner (no bench main) for linking into the plugin.
cuda_library(
    name = "flashinfer_fp4_runner",
    srcs = ["flashinfer_gemm/fp4_flashinfer_runner.cu.cc"],
    hdrs = glob(["flashinfer_gemm/*.h"]),
    copts = [
        "-DFI_FP4_NO_STREAMK",
        "-DFI_FP4_FULL_TILESET",
        "-DFI_FP4_BENCH_NO_MAIN",
    ],
    includes = ["flashinfer_gemm"],
    tags = ["cuda-only", "gpu"],
    deps = [
        "@cutlass4_archive//:cutlass4",
        "@local_config_cuda//cuda:cuda_headers",
    ],
    alwayslink = True,
)

# Production kernel (no test main), for linking into the plugin via the FFI lib.
cuda_library(
    name = "cutlass_fp4_kernel",
    srcs = ["cutlass_fp4_mm.cu.cc"],
    copts = ["-DCUTLASS_FP4_MM_NO_MAIN"],
    tags = ["cuda-only", "gpu"],
    deps = [
        ":flashinfer_fp4_runner",
        "@cutlass4_archive//:cutlass4",
        "@local_config_cuda//cuda:cuda_headers",
    ],
    alwayslink = True,
)

# ---- Vendored FlashInfer SM120 NVFP4 W4A4 autotuned GEMM ----
# Runner + kernel instantiations. FI_FP4_NO_STREAMK halves kernel count; add
# -DFI_FP4_FULL_TILESET to the copts for the full 8-tile autotune ladder, omit
# it for the fast 2-kernel vendoring shakeout.
cuda_library(
    name = "flashinfer_fp4_bench_lib",
    srcs = ["flashinfer_gemm/fp4_flashinfer_runner.cu.cc"],
    hdrs = glob(["flashinfer_gemm/*.h"]),
    copts = [
        "-DFI_FP4_NO_STREAMK",
        "-DFI_FP4_FULL_TILESET",
    ],
    includes = ["flashinfer_gemm"],
    tags = ["cuda-only", "gpu"],
    deps = [
        "@cutlass4_archive//:cutlass4",
        "@local_config_cuda//cuda:cuda_headers",
    ],
    alwayslink = True,
)

# Oracle kernel for the bench: same source as cutlass_fp4_kernel but WITHOUT the
# flashinfer_fp4_runner dep (the bench_lib already provides the FI symbols; a
# second copy via the runner would be a duplicate-symbol link error). Its
# flashinfer_fp4_* externs resolve against bench_lib at final link.
cuda_library(
    name = "cutlass_fp4_kernel_bench",
    srcs = ["cutlass_fp4_mm.cu.cc"],
    copts = ["-DCUTLASS_FP4_MM_NO_MAIN"],
    tags = ["cuda-only", "gpu"],
    deps = [
        "@cutlass4_archive//:cutlass4",
        "@local_config_cuda//cuda:cuda_headers",
    ],
    alwayslink = True,
)

# Standalone bench: autotune FlashInfer over the real decode shapes, correctness
# checked against our CPU-verified cutlass_fp4_mm_sm120a.
cc_binary(
    name = "flashinfer_fp4_bench",
    tags = ["cuda-only", "gpu"],
    deps = [
        ":flashinfer_fp4_bench_lib",
        ":cutlass_fp4_kernel_bench",
        "@local_config_cuda//cuda:cudart",
    ],
)

# XLA FFI handler registering "cutlass$fp4_mm" -> the sm120a kernel.
cc_library(
    name = "cutlass_fp4_ffi",
    srcs = ["cutlass_fp4_ffi.cc"],
    tags = ["cuda-only", "gpu"],
    deps = [
        ":cutlass_fp4_kernel",
        "//xla:xla_data_proto_cc",
        "//xla/backends/gpu:ffi",
        "//xla/ffi",
        "//xla/ffi:ffi_api",
        "//xla/stream_executor:stream",
        "@com_google_absl//absl/status",
        "@local_config_cuda//cuda:cuda_headers",
    ],
    alwayslink = True,
)
