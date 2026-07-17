# Description:
#   MLX is an array framework for Apple silicon. XLA's Metal backend vendors its
#   Metal kernel sources (Steel GEMM, quantized matmul/matvec, sort) -- and only
#   those. Nothing in this repository is built by Bazel: these are .h Metal
#   sources that //xla/service/gpu:metal_include_root embeds verbatim, writes
#   back out once per process, and hands to the Metal compiler as an -I include
#   root. Our own entry sources #include them, so the compiler reads upstream's
#   bytes directly and "verbatim" holds by construction rather than holding only
#   as long as some generator stays correct.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # MIT

exports_files(["LICENSE"])

# The only thing that enters the build graph and the RBE merkle tree: ~88 files,
# ~600 KB, out of a 4.3 MB archive. Deliberately NOT glob(["**"]) -- the C++,
# CUDA and Python trees are dead weight in every remote action's input set.
#
# The whole subtree is one filegroup because MLX's includes are repo-root-
# relative and resolve across it (kernels/utils.h -> steel/utils.h -> ...); a
# per-file dependency list would be a hand-maintained copy of the include DAG,
# i.e. exactly the thing this project exists to delete.
filegroup(
    name = "metal_kernel_headers",
    srcs = glob(["mlx/backend/metal/kernels/**/*.h"]),
)

# Single-file filegroup whose $(location) embed_tree strips a known suffix off
# to recover the archive root -- the root, not the kernels dir, because MLX's
# quoted includes are repo-root-relative (its own AOT build passes -I<repo
# root>), which is also why strip_prefix in workspace.bzl keeps the
# mlx/backend/metal/kernels/ prefix. A filegroup with many files has no single
# $(location), hence the sentinel.
filegroup(
    name = "metal_kernels_root_sentinel",
    srcs = ["mlx/backend/metal/kernels/utils.h"],
)
