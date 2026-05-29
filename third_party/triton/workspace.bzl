"""Provides the repository macro to import Triton."""

def repo():
    """Imports Triton."""

    native.local_repository(
        name = "triton",
        path = "/home/steeve/intel-xpu-backend-for-triton",
    )
