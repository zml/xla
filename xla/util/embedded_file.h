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

#ifndef XLA_UTIL_EMBEDDED_FILE_H_
#define XLA_UTIL_EMBEDDED_FILE_H_

#include "absl/strings/string_view.h"

namespace xla {

// One file embedded by the `embed_tree` rule (xla/util/build_defs.bzl), keyed
// by its path relative to the embedded tree's root.
//
// This is the difference from `embed_files`, which names its accessor after a
// source's basename and discards the path: a tree whose files include each
// other by relative path needs those paths preserved, both because they are the
// key the compiler resolves an #include against and because basenames collide
// across a real directory layout.
struct EmbeddedFile {
  // Root-relative path, e.g. "mlx/backend/metal/kernels/steel/utils.h".
  absl::string_view path;
  // The file's bytes, verbatim.
  absl::string_view contents;
};

}  // namespace xla

#endif  // XLA_UTIL_EMBEDDED_FILE_H_
