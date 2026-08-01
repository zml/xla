# Description:
#   metalBLAS is a Metal GEMM/GEMV library. XLA's Metal backend vendors five of
#   its shader headers -- and only those. Nothing in this repository is built by
#   Bazel: these are `.h` Metal sources that
#   //xla/service/gpu/metal_kernels:metalblas_shaders EMBEDS via xxd (one
#   get_<stem>() string accessor per file), and metalblas_gemm.cc concatenates
#   get_mb_epi() + one family header at RUNTIME with the matching
#   -DMB_BUILD_<NAME>, then hands the string to the Metal compiler. The compiler
#   reads upstream's bytes directly (mpp_tensor.h reproduces as upstream + the
#   MB_TOKCLAMP patch in series.bzl), so "verbatim" holds by construction rather
#   than holding only as long as some generator stays correct.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # MIT

exports_files(["LICENSE"])

# The only thing that enters the build graph and the RBE merkle tree: five
# headers. Deliberately NOT glob(["metalblas/shaders/*.h"]) -- upstream ships 16
# shader `.h` plus the metalblas.metal binder, and only these five ever carry an
# MB_BUILD_* flag: mpp_tensor / gemv_bt / gemv_nt / gemv_t are the four families
# some caller compiles (metalblas_gemm.cc:214, 390, 434, 439), and mb_epi.h is
# the always-prepended epilogue. The other 11 upstream shaders (mpp_gemm,
# simd_gemm, int_gemm, splitk, cgemv_t, cgemv_nt, complex_pack, conv1x1, flipt,
# mpp_sgpipe, and the metalblas.metal binder) are uncompiled dead weight that a
# recent dead-code sweep removed from this bundle; a glob would drag them back
# into every remote action's input set. Keep the list explicit.
filegroup(
    name = "compiled_shaders",
    srcs = [
        "metalblas/shaders/mb_epi.h",
        "metalblas/shaders/mpp_tensor.h",
        "metalblas/shaders/gemv_bt.h",
        "metalblas/shaders/gemv_nt.h",
        "metalblas/shaders/gemv_t.h",
    ],
    visibility = ["//visibility:public"],
)
