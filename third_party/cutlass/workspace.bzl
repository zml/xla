# Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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
# =============================================================================

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "cutlass_archive",
        build_file = "//third_party:cutlass.BUILD",
        # 4.8.0 (not 3.8.0): the SM100/SM120 blockwise-scaled collectives that
        # serve a block-128 FP8 weight arrived after 3.8. Nothing in the tree
        # compiled CUTLASS device code before, so this pin had no other user.
        sha256 = "c72a69301543f9fbe105a308df0936b91322ee4b979658e496dc94941681e504",
        strip_prefix = "cutlass-4.8.0dev",
        urls = tf_mirror_urls("https://github.com/NVIDIA/cutlass/archive/refs/tags/v4.8.0dev.zip"),
    )
