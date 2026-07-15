"""Configurations for StreamExecutor builds"""

load(
    "@local_config_rocm//rocm:build_defs.bzl",
    _if_cuda_or_rocm = "if_cuda_or_rocm",
)

def stream_executor_friends():
    return ["//..."]

def stream_executor_gpu_friends():
    return ["//..."]

def stream_executor_internal():
    return ["//..."]

def tf_additional_cuda_platform_deps():
    return []

def tf_additional_cudnn_plugin_copts():
    return ["-DNV_CUDNN_DISABLE_EXCEPTION"]

# Returns whether any GPU backend is configured.
def if_gpu_is_configured(if_true, if_false = []):
    return select({
        "//xla/tsl:is_cuda_enabled": if_true,
        "@local_config_rocm//rocm:using_hipcc": if_true,
        "@rules_ml_toolchain//common:is_sycl_enabled": if_true,
        "@local_config_musa//musa:using_musa": if_true,
        "//conditions:default": if_false,
    })

def if_cuda_or_rocm(if_true, if_false = []):
    return _if_cuda_or_rocm(if_true, if_false)

def if_cuda_or_rocm_or_musa(if_true, if_false = []):
    return select({
        "//xla/tsl:is_cuda_enabled": if_true,
        "@local_config_rocm//rocm:using_hipcc": if_true,
        "@local_config_musa//musa:using_musa": if_true,
        "//conditions:default": if_false,
    })

# Helps differentiate targets for sycl build from other gpu build targets
def if_cuda_or_rocm_is_configured(if_true, if_false = []):
    return _if_cuda_or_rocm(if_true, if_false)

# nvlink is not available via the pip wheels, disable it since it will create
# unnecessary dependency
def tf_additional_gpu_compilation_copts():
    return ["-DTF_DISABLE_NVLINK_BY_DEFAULT"]

def stream_executor_build_defs_bzl_deps():
    return []
