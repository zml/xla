# buildifier: disable=load-on-top
workspace(name = "xla")

new_local_repository(
    name = "local_hrx_runtime",
    build_file_content = """
package(default_visibility = ["//visibility:public"])

filegroup(
    name = "libamdhip64_so",
    srcs = ["libhrx/src/binding/hip/libamdhip64.so"],
)

genrule(
    name = "libamdhip64_so_7_copy",
    srcs = ["libhrx/src/binding/hip/libamdhip64.so"],
    outs = ["libamdhip64.so.7"],
    cmd = "cp $< $@",
)

filegroup(
    name = "libamdhip64_so_7",
    srcs = [":libamdhip64_so_7_copy"],
)

filegroup(
    name = "libhrx_so",
    srcs = ["libhrx/src/libhrx/libhrx.so"],
)

exports_files([
    "libhrx/src/binding/hip/libamdhip64.so",
    "libhrx/src/libhrx/libhrx.so",
])
""",
    path = "/home/hugo/hrx-system/bazel-bin",
)

# Initialize the XLA repository and all dependencies.
#
# The cascade of load() statements and xla_workspace?() calls works around the
# restriction that load() statements need to be at the top of .bzl files.
# E.g. we can not retrieve a new repository with http_archive and then load()
# a macro from that repository in the same file.

load(":workspace4.bzl", "xla_workspace4")

xla_workspace4()

load(":workspace3.bzl", "xla_workspace3")

xla_workspace3()

# Initialize hermetic C++
load("@rules_ml_toolchain//cc/deps:cc_toolchain_deps.bzl", "cc_toolchain_deps")
load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

cc_toolchain_deps()

register_toolchains("@rules_ml_toolchain//cc:linux_x86_64_linux_x86_64")

register_toolchains("@rules_ml_toolchain//cc:linux_x86_64_linux_x86_64_cuda")

register_toolchains("@rules_ml_toolchain//cc:linux_x86_64_linux_x86_64_sycl")

register_toolchains("@rules_ml_toolchain//cc:linux_x86_64_linux_x86_64_rocm")

register_toolchains("@rules_ml_toolchain//cc:linux_aarch64_linux_aarch64")

register_toolchains("@rules_ml_toolchain//cc:linux_aarch64_linux_aarch64_cuda")

HERMETIC_CC_TOOLCHAIN_VERSION = "v3.1.1"

tf_http_archive(
    name = "hermetic_cc_toolchain",
    sha256 = "907745bf91555f77e8234c0b953371e6cac5ba715d1cf12ff641496dd1bce9d1",
    urls = [
        "https://mirror.bazel.build/github.com/uber/hermetic_cc_toolchain/releases/download/{0}/hermetic_cc_toolchain-{0}.tar.gz".format(HERMETIC_CC_TOOLCHAIN_VERSION),
        "https://github.com/uber/hermetic_cc_toolchain/releases/download/{0}/hermetic_cc_toolchain-{0}.tar.gz".format(HERMETIC_CC_TOOLCHAIN_VERSION),
    ],
)

load("@hermetic_cc_toolchain//toolchain:defs.bzl", zig_toolchains = "toolchains")

# Plain zig_toolchains() will pick reasonable defaults. See
# toolchain/defs.bzl:toolchains on how to change the Zig SDK version and
# download URL.
zig_toolchains()

register_toolchains(
    "@zig_sdk//toolchain:linux_amd64_gnu.2.31",
)

# Initialize hermetic Python
load("//third_party/py:python_init_rules.bzl", "python_init_rules")

python_init_rules()

load("@rules_ml_toolchain//py:python_init_repositories.bzl", "python_init_repositories")

python_init_repositories(
    requirements = {
        "3.11": "//:requirements_lock_3_11.txt",
        "3.12": "//:requirements_lock_3_12.txt",
    },
)

load("@rules_ml_toolchain//py:python_register_toolchain.bzl", "python_register_toolchain")

python_register_toolchain()

load("@rules_ml_toolchain//py:python_init_pip.bzl", "python_init_pip")

python_init_pip()

load("@pypi//:requirements.bzl", "install_deps")

install_deps()

load(":workspace2.bzl", "xla_workspace2")

xla_workspace2()

load(":workspace1.bzl", "xla_workspace1")

xla_workspace1()

load(":workspace0.bzl", "xla_workspace0")

xla_workspace0()

load(
    "@rules_ml_toolchain//gpu/cuda:cuda_json_init_repository.bzl",
    "cuda_json_init_repository",
)

cuda_json_init_repository()

load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
)
load(
    "@rules_ml_toolchain//gpu/cuda:cuda_redist_init_repositories.bzl",
    "cuda_redist_init_repositories",
    "cudnn_redist_init_repository",
)
load(
    "@rules_ml_toolchain//gpu/cuda:cuda_redist_versions.bzl",
    "REDIST_VERSIONS_TO_BUILD_TEMPLATES",
)
load("//third_party/cccl:workspace.bzl", "CCCL_3_2_0_DIST_DICT", "CCCL_GITHUB_VERSIONS_TO_BUILD_TEMPLATES")

cuda_redist_init_repositories(
    cuda_redistributions = CUDA_REDISTRIBUTIONS | CCCL_3_2_0_DIST_DICT,
    redist_versions_to_build_templates = REDIST_VERSIONS_TO_BUILD_TEMPLATES | CCCL_GITHUB_VERSIONS_TO_BUILD_TEMPLATES,
)

cudnn_redist_init_repository(
    cudnn_redistributions = CUDNN_REDISTRIBUTIONS,
)

load(
    "@rules_ml_toolchain//gpu/cuda:cuda_configure.bzl",
    "cuda_configure",
)

cuda_configure(name = "local_config_cuda")

load(
    "@rules_ml_toolchain//gpu/nccl:nccl_redist_init_repository.bzl",
    "nccl_redist_init_repository",
)

nccl_redist_init_repository()

load(
    "@rules_ml_toolchain//gpu/nccl:nccl_configure.bzl",
    "nccl_configure",
)

nccl_configure(name = "local_config_nccl")

load(
    "@rules_ml_toolchain//gpu/nvshmem:nvshmem_json_init_repository.bzl",
    "nvshmem_json_init_repository",
)

nvshmem_json_init_repository()

load(
    "@nvshmem_redist_json//:distributions.bzl",
    "NVSHMEM_REDISTRIBUTIONS",
)
load(
    "@rules_ml_toolchain//gpu/nvshmem:nvshmem_redist_init_repository.bzl",
    "nvshmem_redist_init_repository",
)

nvshmem_redist_init_repository(
    nvshmem_redistributions = NVSHMEM_REDISTRIBUTIONS,
)

# This is used for building nightly PJRT wheels.
load("//build_tools/pjrt_wheels:nightly.bzl", "nightly_timestamp_repo")

nightly_timestamp_repo(name = "nightly_timestamp")

# This is used for building release candidate PJRT wheels.
load("//build_tools/pjrt_wheels:release_candidate.bzl", "rc_number_repo")

rc_number_repo(name = "rc_number")
