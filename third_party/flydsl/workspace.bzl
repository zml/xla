"""Loads FlyDSL for XLA's in-process Fly MLIR backend."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    flydsl_commit = "c62159d0c18e232794a1903f192fc094148f4a43"
    tf_http_archive(
        name = "flydsl",
        build_file = "//third_party/flydsl:flydsl.BUILD.bazel",
        patch_file = ["//third_party/flydsl:mlir_property_ref.patch"],
        sha256 = "bb124ad3819dfcd5b7739108d12025d00dbbd14e9f70a9cb33a5405cec5f6d27",
        strip_prefix = "FlyDSL-{commit}".format(commit = flydsl_commit),
        urls = tf_mirror_urls(
            "https://github.com/ROCm/FlyDSL/archive/{commit}.tar.gz".format(
                commit = flydsl_commit,
            ),
        ),
    )
