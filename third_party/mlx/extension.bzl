"""Module extension for MLX."""

load("//third_party/mlx:workspace.bzl", mlx = "repo")

def _mlx_ext_impl(mctx):  # @unused
    mlx()

mlx_ext = module_extension(
    implementation = _mlx_ext_impl,
)
