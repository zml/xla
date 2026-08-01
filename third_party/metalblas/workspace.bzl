"""Provides the repository macro to import metalBLAS shaders."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

# Applied non-fuzzy: a bump that disturbs a hunk aborts the fetch, which is the
# signal to re-derive the patch rather than shrink its context.
METALBLAS_PATCHES = [
    # Clamps the M grid on the GPU from the real prompt length instead of a
    # host-side read that raced the producer of that metadata.
    "//third_party/metalblas:mb_tokclamp.patch",
    # Slices operands in the (64-bit) pointer rather than in the 32-bit-indexed
    # tensor view, which wrapped past 2 GiB.
    "//third_party/metalblas:mb_gt2gib.patch",
]

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

    Do not adopt upstream's Metal build flags -- the greedy golden is the only
    acceptance bar (same discipline as the @mlx pin).
    """

    # Current upstream HEAD; metalBLAS carries no tags, so the pin records the
    # 40-hex. Four of the five vendored shaders (mb_epi, gemv_bt, gemv_nt,
    # gemv_t) are byte-verbatim against this commit; mpp_tensor.h reproduces as
    # upstream + the patches above. Bumping it is a deliberate act with a golden
    # bench attached.
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
        patch_file = METALBLAS_PATCHES,
    )
