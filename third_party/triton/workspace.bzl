"""Provides the repository macro to import Triton."""

load("//third_party:repo.bzl", "tf_mirror_urls")
load("//third_party/triton:common/series.bzl", "common_patch_list")
load("//third_party/triton:oss_only/series.bzl", "oss_only_patch_list")

def _triton_repository_impl(ctx):
    sycl_enabled = ctx.os.environ.get("TF_NEED_SYCL", "").strip()
    if sycl_enabled:
        urls = ctx.attr.intel_urls
        sha256 = ctx.attr.intel_sha256
        strip_prefix = ctx.attr.intel_strip_prefix
        patch_files = ctx.attr.intel_patch_file
    else:
        urls = ctx.attr.urls
        sha256 = ctx.attr.sha256
        strip_prefix = ctx.attr.strip_prefix
        patch_files = ctx.attr.patch_file

    for patch_file in patch_files:
        if patch_file:
            ctx.path(Label(patch_file))

    ctx.download_and_extract(
        url = urls,
        sha256 = sha256,
        stripPrefix = strip_prefix,
    )

    for patch_file in patch_files:
        patch_file = ctx.path(Label(patch_file)) if patch_file else None
        if patch_file:
            ctx.patch(patch_file, strip = 1)

_triton_repository = repository_rule(
    implementation = _triton_repository_impl,
    attrs = {
        "sha256": attr.string(mandatory = True),
        "urls": attr.string_list(mandatory = True),
        "strip_prefix": attr.string(),
        "patch_file": attr.string_list(),
        "intel_sha256": attr.string(mandatory = True),
        "intel_urls": attr.string_list(mandatory = True),
        "intel_strip_prefix": attr.string(),
        "intel_patch_file": attr.string_list(),
    },
    environ = ["TF_NEED_SYCL"],
)

def repo():
    """Imports Triton."""
    if native.existing_rule("triton"):
        return

    TRITON_COMMIT = "609ced5e3f04e55234115524eb734822331a37d7"
    TRITON_SHA256 = "979b9f9fd6a1dc6a69de20f60357c9b9dc0cbfba3b1169280c75351b592e8b05"
    INTEL_XPU_TRITON_COMMIT = "1bb446691bb53c8fdd5c1fbad43ac7ddf5c03893"
    INTEL_XPU_TRITON_SHA256 = "e2a9eb9d060d35d12009bbbbd2a98f0948a1f5e38b0745e2c07ebdabd9cbde38"
    _triton_repository(
        name = "triton",
        sha256 = TRITON_SHA256,
        strip_prefix = "triton-" + TRITON_COMMIT,
        urls = tf_mirror_urls("https://github.com/triton-lang/triton/archive/{}.tar.gz".format(TRITON_COMMIT)),
        patch_file = common_patch_list + oss_only_patch_list,
        intel_sha256 = INTEL_XPU_TRITON_SHA256,
        intel_strip_prefix = "intel-xpu-backend-for-triton-" + INTEL_XPU_TRITON_COMMIT,
        intel_urls = tf_mirror_urls("https://github.com/zml/intel-xpu-backend-for-triton/archive/{}.tar.gz".format(INTEL_XPU_TRITON_COMMIT)),
    )
