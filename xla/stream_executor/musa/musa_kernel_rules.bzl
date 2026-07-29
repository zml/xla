# Copyright 2026 The OpenXLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Build rules for small MUSA device-only kernel artifacts."""

load("@rules_cc//cc:find_cc_toolchain.bzl", "find_cc_toolchain", "use_cc_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")

visibility("private")

def _musa_device_binary_impl(ctx):
    if ctx.attr.architectures != ["mp_21"]:
        fail(
            "MUSA device binaries currently require exactly the qualified " +
            "S80 architecture [\"mp_21\"], got {}".format(
                ctx.attr.architectures,
            ),
        )
    architecture = ctx.attr.architectures[0]

    cc_toolchain = find_cc_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    _, compilation_outputs = cc_common.compile(
        name = ctx.label.name + "_device_compile",
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        srcs = [ctx.file.src],
        user_compile_flags = [
            "-x",
            "musa",
            "--musa-device-only",
            "-Wno-unknown-warning-option",
        ],
    )
    objects = compilation_outputs.objects
    if len(objects) != 1:
        fail("expected exactly one MUSA device bundle, got {}".format(len(objects)))

    arguments = ctx.actions.args()
    arguments.add("-unbundle")
    arguments.add("-type=o")
    arguments.add("-targets=musa-mtgpu-mt-musa--{}".format(architecture))
    arguments.add("-inputs={}".format(objects[0].path))
    arguments.add("-outputs={}".format(ctx.outputs.out.path))
    ctx.actions.run(
        executable = ctx.file._clang_offload_bundler,
        arguments = [arguments],
        inputs = depset(
            direct = [objects[0]],
            transitive = [ctx.attr._musa_toolchain_data[DefaultInfo].files],
        ),
        outputs = [ctx.outputs.out],
        mnemonic = "MusaDeviceUnbundle",
        progress_message = "Unbundling MUSA device binary %{label}",
    )
    return [DefaultInfo(files = depset([ctx.outputs.out]))]

musa_device_binary = rule(
    implementation = _musa_device_binary_impl,
    attrs = {
        "src": attr.label(
            allow_single_file = [".cc"],
            mandatory = True,
        ),
        "out": attr.output(mandatory = True),
        "architectures": attr.string_list(mandatory = True),
        "_clang_offload_bundler": attr.label(
            allow_single_file = True,
            cfg = "exec",
            default = Label("@local_config_musa//musa:clang_offload_bundler"),
        ),
        "_musa_toolchain_data": attr.label(
            default = Label("@local_config_musa//musa:toolchain_data"),
        ),
        # TODO(b/394414401): Remove this legacy toolchain fallback together
        # with the corresponding fallback in cc_embed_data.
        "_cc_toolchain": attr.label(
            default = Label("@bazel_tools//tools/cpp:current_cc_toolchain"),
        ),
    },
    fragments = ["cpp"],
    toolchains = use_cc_toolchain(),
)
