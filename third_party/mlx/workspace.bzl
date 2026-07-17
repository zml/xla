"""Provides the repository macro to import MLX."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")
load("//third_party/mlx:series.bzl", "mlx_patch_list")

def repo():
    """Imports MLX.

    Only mlx/backend/metal/kernels/**/*.h enters the build graph (see mlx.BUILD).
    Those headers are embedded verbatim and handed to the Metal compiler as an -I
    include root at runtime (//xla/service/gpu:metal_include_root); our own entry
    sources under xla/service/gpu/metal_kernels/mlx_entries/ #include them.
    Nothing here is compiled by Bazel -- MLX's kernels are Metal sources compiled
    at runtime by the thunk that uses them.

    READ third_party/mlx/series.bzl BEFORE BUMPING. In particular: do not adopt
    upstream's -fno-fast-math.
    """

    # v0.32.1. Recovered, not chosen: this is the commit that regenerates the
    # already-checked-in vendored blobs byte-identically, which is how the base
    # of our previously-unrecorded fork was established. Bumping it is a
    # deliberate act with a golden bench attached -- see the BUMPING THE PIN
    # section of series.bzl for what it has to clear.
    MLX_COMMIT = "57c66cac7cb3e5b1eb350488a61f1506b40d39f8"
    MLX_SHA256 = "7ab4250b1a306d81af5cb316c73218bfb120a1202ee07dbf8274c24683195f0a"

    tf_http_archive(
        name = "mlx",
        sha256 = MLX_SHA256,
        # Strips only the mlx-<sha>/ wrapper. The mlx/backend/metal/kernels/
        # prefix underneath is LOAD-BEARING: MLX's quoted includes are
        # repo-root-relative (its own AOT build passes -I<repo root>), so our
        # -I root has to be the archive root, not the kernels dir.
        strip_prefix = "mlx-" + MLX_COMMIT,
        urls = tf_mirror_urls("https://github.com/ml-explore/mlx/archive/{}.tar.gz".format(MLX_COMMIT)),
        build_file = "//third_party/mlx:mlx.BUILD",
        patch_file = mlx_patch_list,
    )
