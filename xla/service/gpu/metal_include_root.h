/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_SERVICE_GPU_METAL_INCLUDE_ROOT_H_
#define XLA_SERVICE_GPU_METAL_INCLUDE_ROOT_H_

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

// The vendored third-party Metal include tree (today: MLX's kernel headers,
// pinned in //third_party/mlx:workspace.bzl) that our own entry sources under
// metal_kernels/mlx_entries/ #include.
//
// The tree is embedded verbatim and written back out to a content-addressed
// directory on first use, and the Metal compiler is pointed at it with -I. The
// point is that upstream's bytes reach the compiler untouched: no build step
// rewrites them, so "verbatim" is a property of the mechanism rather than a
// property of a generator being correct. It is also why nothing regenerates a
// checked-in copy that a human could quietly edit.

namespace xla {
namespace gpu {

// Returns the directory to pass to the Metal compiler as -I, materializing the
// embedded tree on the first call (thereafter cached, including the failure).
// Thread-safe, and safe against a concurrent process racing on the same path:
// the directory is named by the tree's content hash and published with a single
// rename(2), so a reader either sees a complete tree or no tree at all.
absl::StatusOr<std::string> MetalIncludeRoot();

// A stable hash of the embedded tree's contents. Every metallib cache key must
// include this: the compile depends on the include tree just as much as on the
// source text, and with -I the source no longer contains the tree.
//
// It also enforces the design rule that no call site passes an -I of its own --
// two identical translation units resolved against different roots must not
// share a cache entry, and this is what keeps that from happening silently.
absl::string_view MetalIncludeTreeHash();

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_METAL_INCLUDE_ROOT_H_
