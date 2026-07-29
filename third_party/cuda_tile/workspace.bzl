"""Provides the repository macro to import NVIDIA's CUDA Tile IR (cuda-tile).

cuda-tile ships the `cuda_tile` MLIR dialect plus the CUDA Tile IR bytecode
writer/reader. XLA uses it to emit CUDA Tile IR bytecode for the NVIDIA Tile IR
backend. v13.3.3 builds against XLA's LLVM pin as-is.
"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports cuda-tile."""

    CUDA_TILE_VERSION = "13.3.3"
    CUDA_TILE_SHA256 = "92f64f8b6497c3c5d0654a71b2d144278190725b86805b79368599f41a41b48f"

    tf_http_archive(
        name = "cuda_tile",
        build_file = "//third_party:cuda_tile.BUILD",
        patch_file = ["//third_party/cuda_tile:bytecode_inc_paths.patch"],
        sha256 = CUDA_TILE_SHA256,
        strip_prefix = "cuda-tile-" + CUDA_TILE_VERSION,
        urls = tf_mirror_urls(
            "https://github.com/NVIDIA/cuda-tile/archive/refs/tags/v{}.tar.gz".format(CUDA_TILE_VERSION),
        ),
    )
