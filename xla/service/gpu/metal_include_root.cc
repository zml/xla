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

#include "xla/service/gpu/metal_include_root.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tsl/platform/fingerprint.h"
#include "xla/service/gpu/metal_kernels/mlx_include_tree.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util/embedded_file.h"

namespace xla {
namespace gpu {
namespace {

// The tree, sorted by path so the hash is a true content id: $(locations) order
// is Bazel's, and a reordering must not mint a new directory.
std::vector<EmbeddedFile> SortedTree() {
  absl::Span<const EmbeddedFile> files = get_mlx_include_tree();
  std::vector<EmbeddedFile> sorted(files.begin(), files.end());
  std::sort(sorted.begin(), sorted.end(),
            [](const EmbeddedFile& a, const EmbeddedFile& b) {
              return a.path < b.path;
            });
  return sorted;
}

// Farmhash over every path and its contents, length-delimited so no
// concatenation of one file's bytes into the next path can alias.
std::string ComputeTreeHash() {
  const std::vector<EmbeddedFile> files = SortedTree();
  tsl::Fprint128 h = {0, 0};
  for (const EmbeddedFile& f : files) {
    for (absl::string_view part : {f.path, f.contents}) {
      const tsl::Fprint128 p = tsl::Fingerprint128(part);
      h = tsl::FingerprintCat128(h, {p.low64 ^ part.size(), p.high64});
    }
  }
  return absl::StrFormat("%016x%016x", h.high64, h.low64);
}

std::string TempDir() {
  if (const char* t = std::getenv("TMPDIR"); t != nullptr && *t != '\0') {
    absl::string_view dir(t);
    while (dir.size() > 1 && dir.back() == '/') dir.remove_suffix(1);
    return std::string(dir);
  }
  return "/tmp";
}

// Writes the tree under `staging` and publishes it at `root` with one rename(2).
absl::Status MaterializeTree(const std::string& staging,
                             const std::string& root) {
  tsl::Env* env = tsl::Env::Default();
  for (const EmbeddedFile& f : SortedTree()) {
    const std::string path = absl::StrCat(staging, "/", f.path);
    const std::string::size_type slash = path.find_last_of('/');
    if (slash != std::string::npos) {
      TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(path.substr(0, slash)));
    }
    TF_RETURN_IF_ERROR(tsl::WriteStringToFile(env, path, f.contents));
  }
  // Atomic publish. ENOTEMPTY/EEXIST means another process finished first --
  // its tree hashes to the same name, so its bytes are ours, and the loser just
  // drops its staging copy.
  if (std::rename(staging.c_str(), root.c_str()) != 0) {
    if (!env->FileExists(root).ok()) {
      return absl::InternalError(absl::StrFormat(
          "Could not publish the Metal include tree at %s.", root));
    }
    int64_t undeleted_files, undeleted_dirs;
    env->DeleteRecursively(staging, &undeleted_files, &undeleted_dirs)
        .IgnoreError();
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> MaterializeOnce() {
  const std::string root =
      absl::StrCat(TempDir(), "/xla-metal-inc-", MetalIncludeTreeHash());
  tsl::Env* env = tsl::Env::Default();
  // Content-addressed: an existing directory of this name already holds exactly
  // these bytes, so a bumped pin can never be served a stale tree.
  if (env->FileExists(root).ok()) return root;

  std::string staging;
  if (!env->LocalTempFilename(&staging)) {
    return absl::InternalError(
        "Could not create a Metal include tree staging path.");
  }
  TF_RETURN_IF_ERROR(MaterializeTree(staging, root));
  return root;
}

}  // namespace

absl::string_view MetalIncludeTreeHash() {
  static const std::string* const kHash = new std::string(ComputeTreeHash());
  return *kHash;
}

absl::StatusOr<std::string> MetalIncludeRoot() {
  static absl::once_flag once;
  static absl::StatusOr<std::string>* const kRoot =
      new absl::StatusOr<std::string>();
  absl::call_once(once, [] { *kRoot = MaterializeOnce(); });
  return *kRoot;
}

}  // namespace gpu
}  // namespace xla
