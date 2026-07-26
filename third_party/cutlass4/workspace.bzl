load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "cutlass4_archive",
        build_file = "//third_party:cutlass4.BUILD",
        sha256 = "33f57008ece5fe198b1a03fe5665f90daa01d1ca97bb4cfdaca107c8475a6c95",
        strip_prefix = "cutlass-4.6.1",
        urls = tf_mirror_urls("https://github.com/NVIDIA/cutlass/archive/refs/tags/v4.6.1.zip"),
    )
