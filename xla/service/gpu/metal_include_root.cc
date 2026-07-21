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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <unistd.h>  // confstr, _CS_DARWIN_USER_CACHE_DIR
#endif

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
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

// The parent directory the content-addressed tree is materialized under.
//
// On macOS this is the per-user *cache* directory (_CS_DARWIN_USER_CACHE_DIR,
// .../C/), NOT the per-user *temp* directory (.../T/, i.e. $TMPDIR). They live
// on the same volume, but the OS prunes idle files from the temp directory by
// access time after a few days while leaving the directory tree behind -- which
// silently guts our include tree out from under a long-lived process (the Metal
// compiler then fails to open a pruned header). The cache directory is not
// reaped that way. Fall back to $TMPDIR / /tmp where the cache dir is
// unavailable; TreeIsComplete() below still makes a reaped tree self-heal there.
std::string CacheParentDir() {
#if defined(__APPLE__)
  char buf[1024];
  const size_t n = confstr(_CS_DARWIN_USER_CACHE_DIR, buf, sizeof(buf));
  if (n > 0 && n <= sizeof(buf)) {
    absl::string_view dir(buf, n - 1);  // n counts the terminating NUL.
    while (dir.size() > 1 && dir.back() == '/') dir.remove_suffix(1);
    if (!dir.empty()) return std::string(dir);
  }
#endif
  return TempDir();
}

// A tree is only usable if every embedded file is actually on disk. Checking the
// directory alone is not enough: a reaper (or an interrupted materialize) can
// leave the directory skeleton with its files deleted.
bool TreeIsComplete(const std::string& root, tsl::Env* env) {
  for (const EmbeddedFile& f : SortedTree()) {
    if (!env->FileExists(absl::StrCat(root, "/", f.path)).ok()) return false;
  }
  return true;
}

// Writes the whole tree under `staging` (files + parent dirs), no publish.
absl::Status WriteTreeInto(const std::string& staging, tsl::Env* env) {
  for (const EmbeddedFile& f : SortedTree()) {
    const std::string path = absl::StrCat(staging, "/", f.path);
    const std::string::size_type slash = path.find_last_of('/');
    if (slash != std::string::npos) {
      TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(path.substr(0, slash)));
    }
    TF_RETURN_IF_ERROR(tsl::WriteStringToFile(env, path, f.contents));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> EnsureMaterialized() {
  tsl::Env* env = tsl::Env::Default();
  const std::string root =
      absl::StrCat(CacheParentDir(), "/xla-metal-inc-", MetalIncludeTreeHash());

  // Content-addressed: a complete directory of this name already holds exactly
  // these bytes. We verify every file is present, not merely the directory, so a
  // partial tree (reaper, interrupted write) is never served.
  if (TreeIsComplete(root, env)) return root;

  // Build a complete tree in a private staging directory.
  std::string staging;
  if (!env->LocalTempFilename(&staging)) {
    return absl::InternalError(
        "Could not create a Metal include tree staging path.");
  }
  TF_RETURN_IF_ERROR(WriteTreeInto(staging, env));

  // Clear a stale partial tree occupying `root` so the rename can land. We only
  // reach here when `root` is incomplete, so this cannot delete a complete tree
  // another process depends on.
  if (env->FileExists(root).ok()) {
    int64_t undeleted_files, undeleted_dirs;
    env->DeleteRecursively(root, &undeleted_files, &undeleted_dirs)
        .IgnoreError();
  }
  // Atomic publish. On failure the loser adopts the winner's identical tree.
  if (std::rename(staging.c_str(), root.c_str()) == 0) return root;
  if (TreeIsComplete(root, env)) {
    int64_t undeleted_files, undeleted_dirs;
    env->DeleteRecursively(staging, &undeleted_files, &undeleted_dirs)
        .IgnoreError();
    return root;
  }
  // The publish lost a race and `root` is still not complete; use our own
  // complete staging copy so this process can compile regardless.
  return staging;
}

}  // namespace

absl::string_view MetalIncludeTreeHash() {
  static const std::string* const kHash = new std::string(ComputeTreeHash());
  return *kHash;
}

absl::StatusOr<std::string> MetalIncludeRoot() {
  // Re-validate on every call (they occur only on a metallib cache miss): a
  // first-call result cannot be trusted for the life of a long-running process,
  // since the tree could have been pruned since. EnsureMaterialized() is cheap
  // when the tree is already intact.
  static absl::Mutex* const mu = new absl::Mutex();
  absl::MutexLock lock(mu);
  return EnsureMaterialized();
}

}  // namespace gpu
}  // namespace xla
