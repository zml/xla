"""Provides the repository macro to import metalBLAS shaders."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")
load("//third_party/metalblas:series.bzl", "metalblas_patch_list")

def repo():
    """Imports metalBLAS's Metal shader headers.

    Only metalblas/shaders/{mb_epi,mpp_tensor,gemv_bt,gemv_nt,gemv_t}.h enter the
    build graph (see metalblas.BUILD). These are EMBEDDED, NOT included via -I.
    metalBLAS shader headers have zero cross-header includes, so none of the
    include-root machinery the @mlx pin needs applies here: there is no
    metal_include_root, no -I, no root sentinel, no embed_tree. The five headers
    are xxd'd into get_<stem>() string accessors by //xla/service/gpu/
    metal_kernels:metalblas_shaders and concatenated at RUNTIME -- get_mb_epi()
    (the always-prepended epilogue) plus exactly one family header, compiled with
    the matching -DMB_BUILD_<NAME>. Nothing here is built by Bazel.

    READ third_party/metalblas/series.bzl BEFORE BUMPING. In particular: do not
    adopt upstream's Metal build flags -- the greedy golden is the only
    acceptance bar (same discipline as the @mlx pin).
    """

    # Current upstream HEAD; metalBLAS carries no tags, so the pin records the
    # 40-hex. Four of the five vendored shaders (mb_epi, gemv_bt, gemv_nt,
    # gemv_t) are byte-verbatim against this commit; mpp_tensor.h reproduces as
    # upstream + the append-only MB_TOKCLAMP patch in series.bzl. Bumping it is a
    # deliberate act with a golden bench attached -- see the BUMPING THE PIN
    # section of series.bzl for what it has to clear.
    METALBLAS_COMMIT = "b4dd324e74bb2958f00edf49a106c46fae197b7b"
    METALBLAS_SHA256 = "dec73390ff50d5e889c93177e89271494c0ecb8254f336187b65aa45b7235d9b"

    tf_http_archive(
        name = "metalblas",
        sha256 = METALBLAS_SHA256,
        # Strips only the metalBLAS-<sha>/ wrapper. The metalblas/shaders/ prefix
        # underneath is NOT load-bearing (contrast @mlx: no quoted includes here,
        # so no -I root to preserve) -- it survives simply because embed_files
        # names each accessor from the basename and ignores the path.
        strip_prefix = "metalBLAS-" + METALBLAS_COMMIT,
        urls = tf_mirror_urls("https://github.com/Isalia20/metalBLAS/archive/{}.tar.gz".format(METALBLAS_COMMIT)),
        build_file = "//third_party/metalblas:metalblas.BUILD",
        patch_file = metalblas_patch_list,
    )
