"""Module extension for metalBLAS."""

load("//third_party/metalblas:workspace.bzl", metalblas = "repo")

def _metalblas_ext_impl(mctx):  # @unused
    metalblas()

metalblas_ext = module_extension(
    implementation = _metalblas_ext_impl,
)
